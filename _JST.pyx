#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free

cdef extern from "gamma.c":
    cdef double jst_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return jst_lgamma(x)


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length - 1
    while imin <= imax:
        imid = imin + ((imax - imin) / 2)
        if value < arr[imid]:
            imax = imid - 1
        else:
            imin = imid + 1
    if imin <= length - 1 :
        return imin
    else:
        return length - 1

def _sample_topics(int[:] nd, int[:, :] ndl, int[:, :, :] ndlz, int[:, :, :] nlzw, 
                int[:, :] nlz, double[:, :] alpha_lz, double[:] alphasum_l, 
                double[:, :, :] beta_lzw, double[:, :] betasum_lz, double[:, :] gamma_dl, 
                double[:] gammasum_d, int[:] DS, int[:] WS,  int[:] LS, int[:] ZS, int[:] IS, 
                double[:] rands):
    cdef int i, j, k, w, z, d, l, z_new, l_new, res
    cdef double r, dist_cum
    cdef int N = DS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_senti = nlz.shape[0]
    cdef int n_topics = nlz.shape[1]
    cdef double eta_sum = 0

    cdef double* dist_sum = <double*> malloc(n_topics * n_senti * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")

    with nogil:
        for i in xrange(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]
            l = LS[i]

            dec(nd[d])
            dec(ndl[d, l])
            dec(ndlz[d, l, z])
            dec(nlzw[l, z, w])
            dec(nlz[l, z])

            dist_cum = 0
            for j in xrange(n_senti):
                for k in xrange(n_topics):
                    # eta is a double so cdivision yields a double
                    dist_cum += (nlzw[j, k, w] + beta_lzw[j, k, w]) / (nlz[j, k] + betasum_lz[j, k]) *(ndlz[d, j, k] + alpha_lz[j, k]) / (ndl[d, j] + alphasum_l[j]) *(ndl[d, j] + gamma_dl[d, j]) / (nd[d] + gammasum_d[d])
                    index = j * n_topics + k
                    dist_sum[index] = dist_cum

            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            res = searchsorted(dist_sum, n_senti * n_topics, r)
            l_new = res / n_topics
            z_new = res % n_topics

            if IS[i] == 1:
                l_new = l
            ZS[i] = z_new
            LS[i] = l_new

            inc(nd[d])
            inc(ndl[d, l_new])
            inc(ndlz[d, l_new, z_new])
            inc(nlzw[l_new, z_new, w])
            inc(nlz[l_new, z_new])

        free(dist_sum)

cpdef double _loglikelihood(int[:] nd, int[:, :] ndl, int[:, :, :] ndlz, int[:, :, :] nlzw, 
                int[:, :] nlz, double alpha, double beta, double ga) nogil:
    cdef int z, d, l, w
    cdef int D = nd.shape[0]
    cdef int n_topics = nlz.shape[1]
    cdef int n_senti = ndl.shape[1]
    cdef int vocab_size = nlzw.shape[2]

    cdef double ll = 0

    with nogil:
        #calculate log p(w|z,l)
        lgamma_beta = lgamma(beta)
        lgamma_alpha = lgamma(alpha)
        lgamma_gamma = lgamma(ga)

        ll = n_senti*n_topics*lgamma(beta*vocab_size) - n_senti*n_topics*vocab_size*lgamma_beta
        for l in xrange(n_senti):
            for z in xrange(n_topics):
                ll -= lgamma(beta*vocab_size + nlz[l, z])
                for w in xrange(vocab_size):
                    ll += lgamma(beta + nlzw[l, z, w])

        #calculate log p(z|l,d)
        ll += n_senti*D*lgamma(alpha*n_topics) - n_senti*D*n_topics*lgamma(alpha)
        for l in xrange(n_senti):
            for d in xrange(D):
                ll -= lgamma(alpha*n_topics + ndl[d, l])
                for z in xrange(n_topics):
                    ll += lgamma(alpha + ndlz[l, d, z])

        #calculate log p(l|d)
        ll += D*lgamma(n_senti*ga) - D*n_senti*lgamma_gamma
        for d in xrange(D):
            ll -= lgamma(n_senti*ga + nd[d])
            for l in xrange(n_senti):
                ll += lgamma(ga + ndl[d, l])

        return ll