import numpy as np


def hardenedLDivide(A, B, brief=True):
    r"""
    Wrapper of :func:`numpy.linalg.lstsq` as a replacement for MATLAB's
    ``A \ B`` operation.

    Parameters
    ----------
    A : (N, K) ndarray
        “Coefficient” matrix.
    B : (N, 1) ndarray
        Ordinate or “dependent variable” values. If ``B`` is two-dimensional,
        the least squares solution is calculated for each of the :math:`K`
        columns of ``B``.
    brief : bool, optional

        if ``True``, return only ``xHat``.

        if ``False``, return ``xHat, resid, rank, s``

    Returns
    -------
    xHat : {(N,), (N, K)} ndarray
        Least-squares solution. If ``B`` is two-dimensional, the solutions
        are in the :math:`K` columns of ``x``.
    resid : {(1,), (K,), (0,)} ndarray
        Sums of squared residuals: Squared Euclidean 2-norm for each column in
        ``B - A @ xHat``. If the rank of ``A`` :math:`< N` or :math:`M <= N`,
        this is an empty array. If ``B`` is 1-dimensional, this is a ``(1,)``
        shape array. Otherwise the shape is ``(K,)``.
    rank : int
        Rank of matrix ``A``.
    s : (min(M, N),) ndarray
        Singular values of ``A``.

    Notes
    -----
    ``xHat`` is minimum norm solution of the LS problem above.

    Documentation modified from :func:`numpy.linalg.lstsq`.
    """
    try:
        xHat, resid, rank, s = np.linalg.lstsq(A, B)
    except (SystemError, np.linalg.LinAlgError) as e:
        xHat = np.NaN * np.ones(A.shape[1:] + B.shape[1:])
        resid = None
        rank = None
        s = None
        print("Caught an exception in 'hardenedLDivide':")
        print(e)
        print("'hardenedLDivide' returning NaNs and Nones")

    if brief:
        return xHat
    else:
        return xHat, resid, rank, s


def is_numeric(z):
    """
    Proposed in https://stackoverflow.com/questions/25127899/what-is-the-best-way-to-check-if-a-variable-is-of-numeric-type-in-python

    Warnings
    --------
    ``is_numeric(True)`` evaluates to ``True``.
    """
    try:
        temp = z + 0
        return True
    except TypeError:
        return False
