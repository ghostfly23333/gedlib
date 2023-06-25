// =========================================================================
/** \file hungarian-lsape.h
 *  \brief Hungarian algorithm for solving the Linear Sum Assignment Problem
 * with Error-correction (LSAPE), aka minimal-cost error-correcting bipartite
 * graph matching, and its dual problem, according to a given edit cost matrix
 * \author Sebastien Bougleux (Normandie Univ, UNICAEN, ENSICAEN, CNRS, GREYC
 * UMR 6072) \author Luc Brun (Normandie Univ, CNRS - ENSICAEN - UNICAEN, GREYC
 * UMR 6072)
 */
// =========================================================================
/* Hungarian algorithm for solving the Linear Sum Assignment
   Problem with Edition (LSAPE)

   authors: Sebastien Bougleux and Luc Brun
   institution: Normandie Univ, CNRS - ENSICAEN - UNICAEN, GREYC UMR 6072

   -----------------------------------------------------------
   This file is part of LSAPE.

   LSAPE is free software: you can redistribute it and/or modify
   it under the terms of the CeCILL-C License. See README for more
   details.

   -----------------------------------------------------------
   content
   -----------------------------------------------------------
   void hungarianLSAPE(C,n+1,m+1,rho,varrho,u,v,forb_assign)

   Compute a solution to the LSAPE.
   Compute an assignment with error-correction between two sets
   U={0,...,n-1} and V={0,...,m-1}, provided
   a (n+1)x(m+1) edit cost matrix C, and such that the
   total cost sum of the assignment is miniminal. The
   last row and the last column of C represent the costs
   of inserting and removing elements, respectively.

   we always have:
      C[n+(m+1)*m] = 0,  u[n] = v[m] = 0

   The resulting assignment is provided as two mappings
   rho:U->V\cup{m} and varrho:V->U\cup{n}.

   - Worst-case time complexity in O(min{n,m}^2max{n,m})
   - Space complexity in O(nm)

   reference: S. Bougleux and L. Brun
              Linear Sum Assignment with Edition
              Technical Report, March 2016
              Normandie Univ, GREYC UMR 6072
*/
// =========================================================================

#ifndef _HUNGARIAN_LSAPE_
#define _HUNGARIAN_LSAPE_

namespace liblsap {

// __global__ void sampleKernel();

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
  DT *v, unsigned short init_type = 1, bool forb_assign = false);

#define DECL_TMPL(DT, IT)                                                      \
  extern template void hungarianLSAPE(                                         \
    const DT *C, const IT &nrows, const IT &ncols, IT *rho, IT *varrho, DT *u, \
    DT *v, unsigned short init_type = 1, bool forb_assign = false)

DECL_TMPL(double, int);

} // namespace liblsap


// -----------------------------------------------------------
#endif
