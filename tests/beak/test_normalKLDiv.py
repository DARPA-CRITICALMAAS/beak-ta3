"""
test_normalKLDiv - Test suite

This code provides the test suite. It can be run through the pytest
unit testing framework.

New bug reports for the code should not be closed without ensuring that
a test in the suite both fails before the fix, and passes after it.
"""

import pytest
import numpy as np

from beak.math.normalKLDiv import normalKLDiv

import sys

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


# class normalKLDivTests:
def test_syntaxSupport():
    """ Ensure that all of the calling syntaxes remain supported """
    # Scalar calls
    mu1 = 1
    Sigma1 = 2
    mu2 = 3
    Sigma2 = 4
    theta1 = np.stack([mu1, Sigma1])
    theta2 = np.stack([mu2, Sigma2])

    val = normalKLDiv(mu1, Sigma1, mu2, Sigma2)
    assert val == normalKLDiv(theta1=theta1, theta2=theta2)
    assert val == normalKLDiv(theta1=[mu1, Sigma1], theta2=[mu2, Sigma2])

    # Multivariate calls
    mu1 = np.array([list(range(1, 4))])
    Sigma1 = [[1, .1, .03], [.1, 2, .2], [.03, .2, 3]]
    mu2 = np.array([list(range(2, 5))])
    Sigma2 = [[2, .02, .05], [.02, 2, .2], [.05, .2, 4]]

    val = normalKLDiv(mu1, Sigma1, mu2, Sigma2)
    assert val == normalKLDiv(theta1=[mu1, Sigma1], theta2=[mu2, Sigma2])


def test_knownValues():
    """     # Compare against known values
    # The following function was used in Maxima to generate analytic solutions
    # normalKLD(m0,S0,m1,S1):=(tracematrix(S1^^(-1).S0)
                                        - log(determinant(S1^^(-1).S0))
                              + transpose(m1-m0).S1^^(-1).(m1-m0)
                              - matrix_size(m0)[1])/2;
    """
    mu1 = 1
    Sigma1 = 2
    mu2 = 3
    Sigma2 = 4

    assert normalKLDiv(mu1, Sigma1, mu2, Sigma2) == np.log(2) / 2 + 1 / 4

    mu1 = [[1], [2]]
    Sigma1 = [[1, .5], [.5, 1]]
    mu2 = [[3], [4]]
    Sigma2 = [[2, 0], [0, 2]]

    assert normalKLDiv(mu1, Sigma1, mu2, Sigma2) == (3 - np.log(3 / 16)) / 2


def test_edgeCases():
    """ Ensure edge cases work """
    assert np.isnan(normalKLDiv(theta1=[np.NaN, 1], theta2=[1, 1]))
    assert np.isnan(normalKLDiv(theta1=[1, np.NaN], theta2=[1, 1]))
    assert np.isnan(normalKLDiv(theta1=[1, 1], theta2=[np.NaN, 1]))
    assert np.isnan(normalKLDiv(theta1=[1, 1], theta2=[1, np.NaN]))


def test_failureHandling():
    # Scalar calls
    mu1 = 1
    Sigma1 = 2
    mu2 = 3
    Sigma2 = 4

    # test wrong number of inputs
    f = lambda: normalKLDiv(mu1, Sigma1, mu2, Sigma2, 3)
    with pytest.raises(TypeError):
        f()

    # test unexpected input type
    Sigma1_string = {str(Sigma1)}
    f = lambda: normalKLDiv(Sigma1_string, Sigma1_string)
    with pytest.raises(TypeError):
        f()

def test_readPackageData():
    test_array = np.load(str(files("beak.data") / "testData.npy"))

    mu1 = test_array[0]
    Sigma1 = test_array[1]
    mu2 = test_array[2]
    Sigma2 = test_array[3]

    assert normalKLDiv(mu1, Sigma1, mu2, Sigma2) == np.log(2) / 2 + 1 / 4
