#include <hungarian-lsape.cuh>

#include <assert.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

namespace liblsap {

/* Hungarian algorithm:
 * 
 * Initialize the slack matrix with the cost matrix, and then work with the
 * slack matrix.
 * 
 * STEP 1: Subtract the row minimum from each row. Subtract the column minimum
 * from each column.
 * 
 * STEP 2: Find a zero of the slack matrix. If there are no starred zeros in its
 * column or row star the zero. Repeat for each zero.
 * 
 * STEP 3: Cover each column with a starred zero. If all the columns are
 * covered then the matching is maximum.
 * 
 * STEP 4: Find a non-covered zero and prime it. If there is no starred zero in
 * the row containing this primed zero, Go to Step 5. Otherwise, cover this row
 * and uncover the column containing the starred zero. Continue in this manner
 * until there are no uncovered zeros left. Save the smallest uncovered value
 * and Go to Step 6.
 * 
 * STEP 5: Construct a series of alternating primed and starred zeros as
 * follows: Let Z0 represent the uncovered primed zero found in Step 4. Let Z1
 * denote the starred zero in the column of Z0(if any). Let Z2 denote the primed
 * zero in the row of Z1(there will always be one). Continue until the series
 * terminates at a primed zero that has no starred zero in its column. Un-star
 * each starred zero of the series, star each primed zero of the series, erase
 * all primes and uncover every row in the matrix. Return to Step 3.
 * 
 * STEP 6: Add the minimum uncovered value to every element of each covered row,
 * and subtract it from every element of each uncovered column.
 * Return to Step 4 without altering any stars, primes, or covered rows.
 */

#define klog2(n)                                                               \
  ((n < 8)                                                                     \
     ? 2                                                                       \
     : ((n < 16)                                                               \
          ? 3                                                                  \
          : ((n < 32)                                                          \
               ? 4                                                             \
               : ((n < 64)                                                     \
                    ? 5                                                        \
                    : ((n < 128)                                               \
                         ? 6                                                   \
                         : ((n < 256)                                          \
                              ? 7                                              \
                              : ((n < 512)                                     \
                                   ? 8                                         \
                                   : ((n < 1024)                               \
                                        ? 9                                    \
                                        : ((n < 2048)                          \
                                             ? 10                              \
                                             : ((n < 4096)                     \
                                                  ? 11                         \
                                                  : ((n < 8192)                \
                                                       ? 12                    \
                                                       : ((n < 16384)          \
                                                            ? 13               \
                                                            : 0))))))))))))

#ifndef DYNAMIC
#define MANAGED __managed__
#define dh_checkCuda checkCuda
#define dh_get_globaltime get_globaltime
#define dh_get_timer_period get_timer_period
#else
#define dh_checkCuda d_checkCuda
#define dh_get_globaltime d_get_globaltime
#define dh_get_timer_period d_get_timer_period
#define MANAGED
#endif

#define kmin(x, y) ((x < y) ? x : y)
#define kmax(x, y) ((x > y) ? x : y)

const int user_n = 127;
const int n = 1 << (klog2(user_n) + 1);   // The size of the cost/pay matrix
const int log2_n = klog2(n);              // log2(n)
const int max_threads_per_block = 1024;   // The maximum number of threads per block

// Number of threads used in steps 3ini, 3, 4ini, 4a, 4b, 5a and 5b (64)
const int n_threads = kmin(n, 64);
// Number of threads used in step 1 and 6 (256)
const int n_threads_reduction = kmin(n, 256);
// Number of blocks used in step 1 and 6 (256)
const int n_blocks_reduction = kmin(n, 256);
// Number of threads used in steps 2 and 6 (512)
const int n_threads_full = kmin(n, 512); 
// Initialization for the random number generator
const int seed = 45345; 

// Number of blocks used in small kernels
const int n_blocks = n / n_threads; 
// Number of blocks used the largest grid sizes
const int n_blocks_full = n * n / n_threads_full; 
// Used to extract the row from tha matrix position index
// Number of columns per block in step 4
const int columns_per_block_step_4 = 512; 
// Number of blocks in step 4 and 2
const int n_blocks_step_4 = kmax(n / columns_per_block_step_4, 1);
const int row_mask = (1 << log2_n) - 1; 
const int nrows = n, ncols = n;
// Number of rows per block in step 1
const int n_rows_per_block = n / n_blocks_reduction;

// The size of a data block
const int data_block_size = columns_per_block_step_4 * n;
const int log2_data_block_size = log2_n + klog2(columns_per_block_step_4);

typedef int data;
#define MAX_DATA INT_MAX
#define MIN_DATA INT_MIN

// Host Variables

struct HungarianCPUContext {
  data h_cost[ncols][nrows];
  int h_column_of_star_at_row[nrows];
  int h_row_of_star_at_column[ncols];
  int h_zeros_vector_size;
  int h_n_matches;
  bool h_found;
  bool h_goto_5;
};

struct HungarianManagedContext {
  int zeros_size;     // The number fo zeros
  int n_matches;      // Used in step 3 to count the number of matches found
  bool goto_5;        // After step 4, goto step 5?
  bool repeat_kernel; // Needs to repeat the step 2 and step 4 kernel?
#if defined(DEBUG) || defined(_DEBUG)
  int n_covered_rows;    // Used in debug mode to check for the
                         // number of covered rows
  int n_covered_columns; // Used in debug mode to check for
                         // the number of covered columns
#endif
};

// Device Variables
struct HungarianGPUContext {
  data slack[nrows * ncols];         // The slack matrix
  data min_in_rows[nrows];           // Minimum in rows
  data min_in_cols[ncols];           // Minimum in columns
  int zeros[nrows * ncols];          // A vector with the position of the
                                     // zeros in the slack matrix
  int zeros_size_b[n_blocks_step_4]; // The number of zeros in block i

  int row_of_star_at_column[ncols]; // A vector that given the column
                                    // j gives the row of the star at
                                    // that column (or -1, no star)
  int column_of_star_at_row[nrows]; // A vector that given the row i
                                    // gives the column of the star
                                    // at that row (or -1, no star)
  int cover_row[nrows];    // A vector that given the row i indicates if
                           // it is covered (1- covered, 0- uncovered)
  int cover_column[ncols]; // A vector that given the column j indicates if it
                           // is covered (1- covered, 0- uncovered)
  int column_of_prime_at_row[nrows]; // A vector that given the row i
                                     // gives the column of the prime
                                     // at that row  (or -1, no prime)
  int row_of_green_at_column[ncols]; // A vector that given the row j
                                     // gives the column of the green
                                     // at that row (or -1, no green)

  data max_in_mat_row[nrows]; // Used in step 1 to stores the maximum in rows
  data
    min_in_mat_col[ncols]; // Used in step 1 to stores the minimums in columns
  data d_min_in_mat_vect[n_blocks_reduction]; // Used in step 6 to stores the
                                              // intermediate results from the
                                              // first reduction kernel
  data d_min_in_mat; // Used in step 6 to store the minimum

  HungarianManagedContext *managed;
};
__shared__ extern data sdata[]; // For access to shared memory

// -------------------------------------------------------------------------------------
// Device code
// -------------------------------------------------------------------------------------

#if defined(DEBUG) || defined(_DEBUG)
__global__ void convergence_check() {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (cover_column[i])
    atomicAdd((int *)&n_covered_columns, 1);
  if (cover_row[i])
    atomicAdd((int *)&n_covered_rows, 1);
}

#endif

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline __device__ cudaError_t d_checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
};

__global__ void init(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nrows) {
    ctx->cover_row[i] = 0;
    ctx->column_of_star_at_row[i] = -1;
  }
  if (i < ncols) {
    ctx->cover_column[i] = 0;
    ctx->row_of_star_at_column[i] = -1;
  }
}

/* STEP 1: Subtract the row minimum from each row. Subtract the column minimum
 * from each column.
 */

__device__ void min_in_rows_warp_reduce(volatile data *sdata, int tid) {
  if (n_threads_reduction >= 64 && n_rows_per_block < 64)
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (n_threads_reduction >= 32 && n_rows_per_block < 32)
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (n_threads_reduction >= 16 && n_rows_per_block < 16)
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (n_threads_reduction >= 8 && n_rows_per_block < 8)
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (n_threads_reduction >= 4 && n_rows_per_block < 4)
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (n_threads_reduction >= 2 && n_rows_per_block < 2)
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

__global__ void calc_min_in_rows(HungarianGPUContext *ctx) {
  __shared__ data
    sdata[n_threads_reduction]; // One temporary result for each thread.

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  // One gets the line and column from the blockID and threadID.
  unsigned int l = bid * n_rows_per_block + tid % n_rows_per_block;
  unsigned int c = tid / n_rows_per_block;
  unsigned int i = c * nrows + l;
  const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
  data thread_min = MAX_DATA;

  while (i < n * n) {
    thread_min = min(thread_min, ctx->slack[i]);
    i += gridSize; // go to the next piece of the matrix...
                   // gridSize = 2^k * n, so that each thread always processes
                   // the same line or column
  }
  sdata[tid] = thread_min;

  __syncthreads();
  if (n_threads_reduction >= 1024 && n_rows_per_block < 1024) {
    if (tid < 512) {
      sdata[tid] = min(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 512 && n_rows_per_block < 512) {
    if (tid < 256) {
      sdata[tid] = min(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 256 && n_rows_per_block < 256) {
    if (tid < 128) {
      sdata[tid] = min(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 128 && n_rows_per_block < 128) {
    if (tid < 64) {
      sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32)
    min_in_rows_warp_reduce(sdata, tid);
  if (tid < n_rows_per_block)
    ctx->min_in_rows[bid * n_rows_per_block + tid] = sdata[tid];
}

// a) Subtracting the column by the minimum in each column
const int n_cols_per_block = n / n_blocks_reduction;

__device__ void min_in_cols_warp_reduce(volatile data *sdata, int tid) {
  if (n_threads_reduction >= 64 && n_cols_per_block < 64)
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (n_threads_reduction >= 32 && n_cols_per_block < 32)
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (n_threads_reduction >= 16 && n_cols_per_block < 16)
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (n_threads_reduction >= 8 && n_cols_per_block < 8)
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (n_threads_reduction >= 4 && n_cols_per_block < 4)
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (n_threads_reduction >= 2 && n_cols_per_block < 2)
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

__global__ void calc_min_in_cols(HungarianGPUContext *ctx) {
  __shared__ data
    sdata[n_threads_reduction]; // One temporary result for each thread

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  // One gets the line and column from the blockID and threadID.
  unsigned int c = bid * n_cols_per_block + tid % n_cols_per_block;
  unsigned int l = tid / n_cols_per_block;
  const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
  data thread_min = MAX_DATA;

  while (l < n) {
    unsigned int i = c * nrows + l;
    thread_min = min(thread_min, ctx->slack[i]);
    l += gridSize / n; // go to the next piece of the matrix...
                       // gridSize = 2^k * n, so that each thread always
                       // processes the same line or column
  }
  sdata[tid] = thread_min;

  __syncthreads();
  if (n_threads_reduction >= 1024 && n_cols_per_block < 1024) {
    if (tid < 512) {
      sdata[tid] = min(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 512 && n_cols_per_block < 512) {
    if (tid < 256) {
      sdata[tid] = min(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 256 && n_cols_per_block < 256) {
    if (tid < 128) {
      sdata[tid] = min(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 128 && n_cols_per_block < 128) {
    if (tid < 64) {
      sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32)
    min_in_cols_warp_reduce(sdata, tid);
  if (tid < n_cols_per_block)
    ctx->min_in_cols[bid * n_cols_per_block + tid] = sdata[tid];
}

__global__ void step_1_row_sub(HungarianGPUContext *ctx) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = i & row_mask;

  ctx->slack[i] =
    ctx->slack[i] -
    ctx->min_in_rows[l]; // subtract the minimum in row from that row
}

__global__ void step_1_col_sub(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int c = i >> log2_n;
  ctx->slack[i] =
    ctx->slack[i] -
    ctx->min_in_cols[c]; // subtract the minimum in row from that row

  if (i == 0)
    ctx->managed->zeros_size = 0;
  if (i < n_blocks_step_4)
    ctx->zeros_size_b[i] = 0;
}

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {
    static __shared__ int shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);

    //write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x<blockDim.x / 32) ? shared[lane] : int(0);
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// Compress matrix
__global__ void compress_matrix(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int pred = ctx->slack[i] == 0;
  int num_zeros = blockReduceSum(pred);
  if (pred) {
    atomicAdd(&ctx->managed->zeros_size, 1);
    // inter-block reduce
    // if (threadIdx.x == 0) {
    //   atomicAdd(&ctx->managed->zeros_size, num_zeros);
    // }
    int b = i >> log2_data_block_size;
    int i0 = i & ~(data_block_size - 1); // == b << log2_data_block_size

    // if (threadIdx.x == 0) {
    //   int j = atomicAdd(ctx->zeros_size_b + b, num_zeros);
    //   ctx->zeros[i0 + j] = i;
    // }
    int j = atomicAdd(ctx->zeros_size_b + b, 1);
    ctx->zeros[i0 + j] = i;
  }
}

/* STEP 2
 * Find a zero of slack. If there are no starred zeros in its
 * column or row star the zero. Repeat for each zero.
 * The zeros are split through blocks of data so we run step 2 with several
 * thread blocks and rerun the kernel if repeat was set to true.
 */
__global__ void step_2(HungarianGPUContext *ctx) {
  int i = threadIdx.x;
  int b = blockIdx.x;
  __shared__ bool repeat;
  __shared__ bool s_repeat_kernel;

  if (i == 0)
    s_repeat_kernel = false;

  do {
    __syncthreads();
    if (i == 0)
      repeat = false;
    __syncthreads();

    for (int j = i; j < ctx->zeros_size_b[b]; j += blockDim.x) {
      int z = ctx->zeros[(b << log2_data_block_size) + j];
      int l = z & row_mask;
      int c = z >> log2_n;

      if (ctx->cover_row[l] == 0 && ctx->cover_column[c] == 0) {
        if (!atomicExch((int *)&(ctx->cover_row[l]), 1)) {
          if (!atomicExch((int *)&(ctx->cover_column[c]), 1)) {
            ctx->row_of_star_at_column[c] = l;
            ctx->column_of_star_at_row[l] = c;
          } else {
            ctx->cover_row[l] = 0;
            repeat = true;
            s_repeat_kernel = true;
          }
        }
      }
    }
    __syncthreads();
  } while (repeat);

  if (s_repeat_kernel)
    ctx->managed->repeat_kernel = true;
}

// STEP 3
// uncover all the rows and columns before going to step 3
__global__ void step_3ini(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  ctx->cover_row[i] = 0;
  ctx->cover_column[i] = 0;
  if (i == 0)
    ctx->managed->n_matches = 0;
}

// Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum
__global__ void step_3(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (ctx->row_of_star_at_column[i] >= 0) {
    ctx->cover_column[i] = 1;
    atomicAdd((int *)&ctx->managed->n_matches, 1);
  }
}

/* STEP 4
 * Find a noncovered zero and prime it. If there is no starred
 * zero in the row containing this primed zero, go to Step 5.
 * Otherwise, cover this row and uncover the column containing
 * the starred zero. Continue in this manner until there are no
 * uncovered zeros left. Save the smallest uncovered value and
 * Go to Step 6.
 */

__global__ void step_4_init(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  ctx->column_of_prime_at_row[i] = -1;
  ctx->row_of_green_at_column[i] = -1;
}

__global__ void step_4(HungarianGPUContext *ctx) {
  __shared__ bool s_found;
  __shared__ bool s_goto_5;
  __shared__ bool s_repeat_kernel;
  volatile int *v_cover_row = ctx->cover_row;
  volatile int *v_cover_column = ctx->cover_column;

  int i = threadIdx.x;
  int b = blockIdx.x;

  if (i == 0) {
    s_repeat_kernel = false;
    s_goto_5 = false;
  }

  do {
    __syncthreads();
    if (i == 0)
      s_found = false;
    __syncthreads();

    for (int j = i; j < ctx->zeros_size_b[b]; j += blockDim.x) {
      int z = ctx->zeros[(b << log2_data_block_size) + j];
      int l = z & row_mask;
      int c = z >> log2_n;
      int c1 = ctx->column_of_star_at_row[l];

      for (int n = 0; n < 10; n++) {

        if (!v_cover_column[c] && !v_cover_row[l]) {
          s_found = true;
          s_repeat_kernel = true;
          ctx->column_of_prime_at_row[l] = c;

          if (c1 >= 0) {
            v_cover_row[l] = 1;
            __threadfence();
            v_cover_column[c1] = 0;
          } else {
            s_goto_5 = true;
          }
        }
      }

    }
    __syncthreads();
  } while (s_found && !s_goto_5);

  if (i == 0 && s_repeat_kernel)
    ctx->managed->repeat_kernel = true;
  if (i == 0 && s_goto_5)
    ctx->managed->goto_5 = true;
}

/* STEP 5:
 * Construct a series of alternating primed and starred zeros as
 * follows:
 * Let Z0 represent the uncovered primed zero found in Step 4.
 * Let Z1 denote the starred zero in the column of Z0(if any).
 * Let Z2 denote the primed zero in the row of Z1(there will always
 * be one). Continue until the series terminates at a primed zero
 * that has no starred zero in its column. Unstar each starred
 * zero of the series, star each primed zero of the series, erase
 * all primes and uncover every line in the matrix. Return to Step 3.
 */

// Eliminates joining paths
__global__ void step_5a(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int r_Z0, c_Z0;

  c_Z0 = ctx->column_of_prime_at_row[i];
  if (c_Z0 >= 0 && ctx->column_of_star_at_row[i] < 0) {
    ctx->row_of_green_at_column[c_Z0] = i;

    while ((r_Z0 = ctx->row_of_star_at_column[c_Z0]) >= 0) {
      c_Z0 = ctx->column_of_prime_at_row[r_Z0];
      ctx->row_of_green_at_column[c_Z0] = r_Z0;
    }
  }
}

// Applies the alternating paths
__global__ void step_5b(HungarianGPUContext *ctx) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  int r_Z0, c_Z0, c_Z2;

  r_Z0 = ctx->row_of_green_at_column[j];

  if (r_Z0 >= 0 && ctx->row_of_star_at_column[j] < 0) {

    c_Z2 = ctx->column_of_star_at_row[r_Z0];

    ctx->column_of_star_at_row[r_Z0] = j;
    ctx->row_of_star_at_column[j] = r_Z0;

    while (c_Z2 >= 0) {
      r_Z0 = ctx->row_of_green_at_column[c_Z2]; // row of Z2
      c_Z0 = c_Z2;                              // col of Z2
      c_Z2 = ctx->column_of_star_at_row[r_Z0];  // col of Z4

      // star Z2
      ctx->column_of_star_at_row[r_Z0] = c_Z0;
      ctx->row_of_star_at_column[c_Z0] = r_Z0;
    }
  }
}

/* STEP 6
 * Add the minimum uncovered value to every element of each covered
 * row, and subtract it from every element of each uncovered column.
 * Return to Step 4 without altering any stars, primes, or covered lines.
 */

template <unsigned int blockSize>
__device__ void min_warp_reduce(volatile data *sdata, int tid) {
  if (blockSize >= 64)
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (blockSize >= 32)
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (blockSize >= 16)
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (blockSize >= 8)
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (blockSize >= 4)
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (blockSize >= 2)
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize> // blockSize is the size of a block of threads
__device__ void min_reduce1(
  HungarianGPUContext *gpu_ctx, volatile data *g_idata, volatile data *g_odata,
  unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = MAX_DATA;

  while (i < n) {
    int i1 = i;
    int i2 = i + blockSize;
    int l1 = i1 & row_mask;
    int c1 = i1 >> log2_n;
    data g1;
    if (gpu_ctx->cover_row[l1] == 1 || gpu_ctx->cover_column[c1] == 1)
      g1 = MAX_DATA;
    else
      g1 = g_idata[i1];
    int l2 = i2 & row_mask;
    int c2 = i2 >> log2_n;
    data g2;
    if (gpu_ctx->cover_row[l2] == 1 || gpu_ctx->cover_column[c2] == 1)
      g2 = MAX_DATA;
    else
      g2 = g_idata[i2];
    sdata[tid] = min(sdata[tid], min(g1, g2));
    i += gridSize;
  }

  __syncthreads();
  if (blockSize >= 1024) {
    if (tid < 512) {
      sdata[tid] = min(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = min(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = min(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32)
    min_warp_reduce<blockSize>(sdata, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void min_reduce2(
  HungarianGPUContext *gpu_ctx, volatile data *g_idata, volatile data *g_odata,
  unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;

  sdata[tid] = min(g_idata[i], g_idata[i + blockSize]);

  __syncthreads();
  if (blockSize >= 1024) {
    if (tid < 512) {
      sdata[tid] = min(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = min(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = min(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32)
    min_warp_reduce<blockSize>(sdata, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

__global__ void step_6_add_sub(HungarianGPUContext *ctx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = i & row_mask;
  int c = i >> log2_n;
  if (ctx->cover_row[l] == 1 && ctx->cover_column[c] == 1)
    ctx->slack[i] += ctx->d_min_in_mat;
  if (ctx->cover_row[l] == 0 && ctx->cover_column[c] == 0)
    ctx->slack[i] -= ctx->d_min_in_mat;

  if (i == 0)
    ctx->managed->zeros_size = 0;
  if (i < n_blocks_step_4)
    ctx->zeros_size_b[i] = 0;
}

__global__ void min_reduce_kernel1(HungarianGPUContext *ctx) {
  min_reduce1<n_threads_reduction>(
    ctx, ctx->slack, ctx->d_min_in_mat_vect, nrows * ncols);
}

__global__ void min_reduce_kernel2(HungarianGPUContext *ctx) {
  min_reduce2<n_threads_reduction / 2>(
    ctx, ctx->d_min_in_mat_vect, &ctx->d_min_in_mat, n_blocks_reduction);
}

__device__ inline long long int d_get_globaltime(void) {
  long long int ret;

  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));

  return ret;
}

// Returns the period in miliseconds
__device__ inline double d_get_timer_period(void) { return 1.0e-6; }

// -------------------------------------------------------------------------------------
// Host code
// -------------------------------------------------------------------------------------

// Convenience function for checking CUDA runtime API results
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
};

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void) {
  using namespace std::chrono;
  return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void) {
  using namespace std::chrono;
  return 1000.0 * high_resolution_clock::period::num /
         high_resolution_clock::period::den;
}

#define declare_kernel(k)                                                      \
  hr_clock_rep k##_time = 0;                                                   \
  static int k##_runs = 0

#define call_kernel(k, n_blocks, n_threads, ctx)                               \
  call_kernel_s(k, n_blocks, n_threads, 0ll, ctx)

#define call_kernel_s(k, n_blocks, n_threads, shared, ctx)                     \
  {                                                                            \
    timer_start = dh_get_globaltime();                                         \
    k<<<n_blocks, n_threads, shared, stream>>>(ctx);                           \
    dh_checkCuda(cudaStreamSynchronize(stream));                               \
    timer_stop = dh_get_globaltime();                                          \
    k##_time += timer_stop - timer_start;                                      \
    k##_runs++;                                                                \
  }
// printf("Finished kernel " #k "(%d,%d,%lld)\n", n_blocks, n_threads, shared);			\
// fflush(0);											\

#define kernel_stats(k)                                                        \
  fprintf(stderr, #k " %010ldns\t(%05d)\n", k##_time, k##_runs);

// Hungarian_Algorithm
void Hungarian_Algorithm(HungarianCPUContext *cpu_ctx, cudaStream_t stream) {
  hr_clock_rep timer_start, timer_stop;
  hr_clock_rep total_time_start, total_time_stop;
#if defined(DEBUG) || defined(_DEBUG)
  int last_n_covered_rows = 0, last_n_matches = 0;
#endif

  HungarianGPUContext *gpu_ctx;
  HungarianManagedContext *managed_ctx;
  checkCuda(cudaMalloc(&gpu_ctx, sizeof(HungarianGPUContext)));
  checkCuda(cudaMallocManaged(&managed_ctx, sizeof(HungarianManagedContext)));

  checkCuda(
    cudaStreamAttachMemAsync(stream, managed_ctx, 0, cudaMemAttachSingle));
  checkCuda(cudaStreamSynchronize(stream));

  cudaMemcpyAsync(
    &gpu_ctx->managed, &managed_ctx, sizeof(managed_ctx),
    cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

  // Copy vectors from host memory to device memory
  cudaMemcpyAsync(
    gpu_ctx->slack, cpu_ctx->h_cost, sizeof(data) * nrows * ncols,
    cudaMemcpyKind::cudaMemcpyHostToDevice,
    stream); // symbol refers to the device
             // memory hence "To" means from
             // Host to Device
  checkCuda(cudaStreamSynchronize(stream));
  // printf("pass memcpy!\n");

  declare_kernel(init);
  declare_kernel(calc_min_in_rows);
  declare_kernel(step_1_row_sub);
  declare_kernel(calc_min_in_cols);
  declare_kernel(step_1_col_sub);
  declare_kernel(compress_matrix);
  declare_kernel(step_2);
  declare_kernel(step_3ini);
  declare_kernel(step_3);
  declare_kernel(step_4_init);
  declare_kernel(step_4);
  declare_kernel(min_reduce_kernel1);
  declare_kernel(min_reduce_kernel2);
  declare_kernel(step_6_add_sub);
  declare_kernel(step_5a);
  declare_kernel(step_5b);
  declare_kernel(step_5c);
  // printf("before declare!\n");
  total_time_start = dh_get_globaltime();

  // Initialization
  call_kernel(init, n_blocks, n_threads, gpu_ctx);

  // Step 1 kernels
  call_kernel(
    calc_min_in_rows, n_blocks_reduction, n_threads_reduction, gpu_ctx);
  call_kernel(step_1_row_sub, n_blocks_full, n_threads_full, gpu_ctx);
  call_kernel(
    calc_min_in_cols, n_blocks_reduction, n_threads_reduction, gpu_ctx);
  call_kernel(step_1_col_sub, n_blocks_full, n_threads_full, gpu_ctx);

  // compress_matrix
  call_kernel(compress_matrix, n_blocks_full, n_threads_full, gpu_ctx);
  // Step 2 kernels
  do {
    managed_ctx->repeat_kernel = false;
    dh_checkCuda(cudaStreamSynchronize(stream));
    // printf("before step2 kernel call!\n");
    call_kernel(
      step_2, n_blocks_step_4,
      (n_blocks_step_4 > 1 ||
       managed_ctx->zeros_size > max_threads_per_block)
        ? max_threads_per_block
        : managed_ctx->zeros_size,
      gpu_ctx);
    // printf("after step2 kernel call!\n");
    // If we have more than one block it means that we have 512 lines per block
    // so 1024 threads should be adequate.
  } while (managed_ctx->repeat_kernel);
  // printf("before step3!\n");
  while (1) { // repeat steps 3 to 6

    // Step 3 kernels
    call_kernel(step_3ini, n_blocks, n_threads, gpu_ctx);
    call_kernel(step_3, n_blocks, n_threads, gpu_ctx);

    if (managed_ctx->n_matches >= ncols)
      break; // It's done

    // step 4_kernels
    call_kernel(step_4_init, n_blocks, n_threads, gpu_ctx);

    while (1) // repeat step 4 and 6
    {
#if defined(DEBUG) || defined(_DEBUG)
      // At each iteraton either the number of matched or covered rows has to
      // increase. If we went to step 5 the number of matches increases. If we
      // went to step 6 the number of covered rows increases.
      n_covered_rows = 0;
      n_covered_columns = 0;
      dh_checkCuda(cudaStreamSynchronize(stream));
      // really?
      convergence_check<<<n_blocks, n_threads, stream>>>();
      dh_checkCuda(cudaStreamSynchronize(stream));
      assert(
        n_matches > last_n_matches || n_covered_rows > last_n_covered_rows);
      assert(n_matches == n_covered_columns + n_covered_rows);
      last_n_matches = n_matches;
      last_n_covered_rows = n_covered_rows;
#endif
      do { // step 4 loop
        managed_ctx->goto_5 = false;
        managed_ctx->repeat_kernel = false;
        dh_checkCuda(cudaStreamSynchronize(stream));

        call_kernel(
          step_4, n_blocks_step_4,
          (n_blocks_step_4 > 1 ||
           managed_ctx->zeros_size > max_threads_per_block)
            ? max_threads_per_block
            : managed_ctx->zeros_size,
          gpu_ctx);
        // If we have more than one block it means that we have 512 lines per
        // block so 1024 threads should be adequate.

      } while (managed_ctx->repeat_kernel && !managed_ctx->goto_5);

      if (managed_ctx->goto_5)
        break;

      // step 6_kernel
      call_kernel_s(
        min_reduce_kernel1, n_blocks_reduction, n_threads_reduction,
        n_threads_reduction * sizeof(int), gpu_ctx);
      call_kernel_s(
        min_reduce_kernel2, 1, n_blocks_reduction / 2,
        (n_blocks_reduction / 2) * sizeof(int), gpu_ctx);
      call_kernel(step_6_add_sub, n_blocks_full, n_threads_full, gpu_ctx);

      // compress_matrix
      call_kernel(compress_matrix, n_blocks_full, n_threads_full, gpu_ctx);

    } // repeat step 4 and 6

    call_kernel(step_5a, n_blocks, n_threads, gpu_ctx);
    call_kernel(step_5b, n_blocks, n_threads, gpu_ctx);

  } // repeat steps 3 to 6

  checkCuda(cudaStreamSynchronize(stream));
  // printf("ready to copy back!\n");
  // Copy assignments from Device to Host and calculate the total Cost
  cudaMemcpyAsync(
    cpu_ctx->h_column_of_star_at_row, gpu_ctx->column_of_star_at_row,
    nrows * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(
    cpu_ctx->h_row_of_star_at_column, gpu_ctx->row_of_star_at_column,
    ncols * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
  checkCuda(cudaStreamSynchronize(stream));
  kernel_stats(init);
  kernel_stats(calc_min_in_rows);
  kernel_stats(step_1_row_sub);
  kernel_stats(calc_min_in_cols);
  kernel_stats(step_1_col_sub);
  kernel_stats(compress_matrix);
  kernel_stats(step_2);
  kernel_stats(step_3ini);
  kernel_stats(step_3);
  kernel_stats(step_4_init);
  kernel_stats(step_4);
  kernel_stats(min_reduce_kernel1);
  kernel_stats(min_reduce_kernel2);
  kernel_stats(step_6_add_sub);
  kernel_stats(step_5a);
  kernel_stats(step_5b);
  kernel_stats(step_5c);
  // printf("ready to free!\n");
  cudaFree(gpu_ctx);
  cudaFree(managed_ctx);
}

// -----------------------------------------------------------
// Main function: Hungarian algorithm for LSAPE
// -----------------------------------------------------------
/**
 * \brief Compute a solution to the LSAPE (minimal-cost error-correcting
 * bipartite graph matching) with the Hungarian method \param[in] C nrowsxncols
 * edit cost matrix represented as an array if size \p nrows.ncols obtained by
 * concatenating its columns, column \p nrows-1 are the costs of removing the
 * elements of the 1st set, and the row \p ncols-1 represents the costs of
 * inserting an element of the 2nd set \param[in] nrows Number of rows of \p C
 * \param[in] ncols Number of columns of \p C
 * \param[out] rho Array of size \p nrows-1 (must be previously allocated),
 * rho[i]=j indicates that i is assigned to j (substituted by j if j<ncols-1, or
 * removed if j=ncols-1) \param[out] varrho Array of size \p m (must be
 * previously allocated), varrho[j]=i indicates that j is assigned to i
 * (substituted to i if i<nrows-1, or inserted if i=nrows) \param[out] u Array
 * of dual variables associated to the 1st set (rows of \p C), of size \p nrows
 * \param[out] v Array of dual variables associated to the 2nd set (columns of
 * \p C), of size \p ncols \param[in] forb_assign If true, forbidden assignments
 * are marked with negative values in the cost matrix \details A solution to the
 * LSAPE is computed with the primal-dual version of the Hungarian algorithm, as
 * detailed in: \li <em>S. Bougleux and L. Brun, Linear Sum Assignment with
 * Edition, Technical Report, Normandie Univ, GREYC UMR 6072, 2016</em>
 *
 * This version updates dual variables \c u and \c v, and at each iteration, the
 * current matching is augmented by growing only one Hungarian tree until an
 * augmenting path is found. Our implementation uses a Bread-First-like strategy
 * to construct the tree, according to a FIFO strategy to select the next
 * element at each iteration of the growing process.
 *
 * Complexities:
 * \li O(min{n,m}²max{n,m}) in time (worst-case)
 * \li O(nm) in space
 *
 * \remark
 * Template \p DT allows to compute a solution with integer or floating-point
 * values. Note that rounding errors may occur with floating point values when
 * dual variables are updated but this does not affect the overall process.
 */
template <class DT, typename IT>
void hungarianLSAPE(
  const DT *C, const IT &nrows, const IT &ncols, IT *rho, IT *varrho, DT *u,
  DT *v, unsigned short init_type, bool forb_assign) {
  // printf("successful enter hungarianLSAPE\n");
  const IT n = nrows - 1, m = ncols - 1;

  HungarianCPUContext ctx{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      ctx.h_cost[i][j] = C[j * nrows + i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = n; j < ::liblsap::n; j++)
      if (j - n == i) {
        ctx.h_cost[i][j] = C[n * nrows + i];
      } else {
        ctx.h_cost[i][j] = MAX_DATA;
      }
  }
  for (int i = 0; i < m; i++) {
    for (int j = n; j < ::liblsap::n; j++)
      if (j - n == i) {
        ctx.h_cost[j][i] = C[i * nrows + m];
      } else {
        ctx.h_cost[j][i] = MAX_DATA;
      }
  }
  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream));

  static int t_count = 0;
  auto t_iter_start = std::chrono::high_resolution_clock::now();
  Hungarian_Algorithm(&ctx, stream);
  auto used = (std::chrono::high_resolution_clock::now() - t_iter_start).count();
  fprintf(stderr, "used %010ldns\t(%05d)\n", used, t_count++);

  checkCuda(cudaStreamDestroy(stream));

  for (int i = 0; i < n; i++) {
    rho[i] = std::min(ctx.h_column_of_star_at_row[i], m);
  }

  for (int i = 0; i < m; i++) {
    varrho[i] = std::min(ctx.h_row_of_star_at_column[i], n);
  }
}

#define DEFINE_TMPL(DT, IT)                                                    \
  template void hungarianLSAPE(                                                \
    const DT *C, const IT &nrows, const IT &ncols, IT *rho, IT *varrho, DT *u, \
    DT *v, unsigned short init_type, bool forb_assign)

DEFINE_TMPL(double, int);

} // namespace liblsap
