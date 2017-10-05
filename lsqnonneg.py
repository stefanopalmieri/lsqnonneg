"""
A Python implementation of NNLS algorithm

References:
[1]  Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974.

"""

import numpy

def lsqnonneg(C, d):
    '''Linear least squares with nonnegativity constraints.

    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real
    '''

    eps = 2.22e-16    # from matlab

    tol = 10*eps*numpy.linalg.norm(C,1)*(max(C.shape)+1)

    C = numpy.asarray(C)

    (m,n) = C.shape
    P = []
    R = [x for x in range(0,n)]

    x = numpy.zeros(n)

    resid = d - numpy.dot(C, x)
    w = numpy.dot(C.T, resid)

    count = 0

    # outer loop to put variables into set to hold positive coefficients
    while numpy.any(R) and numpy.max(w) > tol:

        j = numpy.argmax(w)
        P.append(j)
        R.remove(j)


        AP = numpy.zeros(C.shape)
        AP[:,P] = C[:,P]

        s=numpy.dot(numpy.linalg.pinv(AP), d)

        s[R] = 0
     
        while numpy.min(s) < 0:


            i = [i for i in P if s[i] <= 0]

            alpha = min(x[i]/(x[i] - s[i]))
            x = x + alpha*(s-x)

            j = [j for j in P if x[j] == 0]
            if j:
                R.append(*j)
                P.remove(j)
            
            AP = numpy.zeros(C.shape)
            AP[:,P] = C[:,P]
            s=numpy.dot(numpy.linalg.pinv(AP), d)
            s[R] = 0
     
        x = s
        resid = d - numpy.dot(C, x)

        w = numpy.dot(C.T, resid)



    return (x, sum(resid * resid), resid)

if __name__=='__main__':

    C = numpy.array([[1, 0, 0.035, 0.09,  0, 0.125, 0.875],
                 [1, 0, 0.01,  0.95,  0, 0.960, 0.040],
                 [1, 0, 0.35,  0.058, 0, 0.408, 0.592],
                 [1, 1, 0,         0, 0, 1.000, 0.000],
                 [1, .96, 0,  0, 0, 0.96, 0.04],
                 [1, .40, 0,  0.40, 0.14, 0.94, 0.06]])

    d = numpy.array([1530, 278, 92.3, 150.5, 7, 527.8, 1002.2])

    [x, resnorm, residual] = lsqnonneg(C.T, d)

    print (x)

    print(numpy.dot(C.T, x))

    C1 = numpy.array([[0.0372, 0.2869, 0.4],
                      [0.6861, 0.7071, 0.3],
                      [0.6233, 0.6245, 0.1],
                      [0.6344, 0.6170, 0.5]])

    d = numpy.array([0.8587, 0.1781, 0.0747, 0.8405])

    [x, resnorm, residual] = lsqnonneg(C1, d)

    print (x)