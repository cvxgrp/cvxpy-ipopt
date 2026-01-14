import cvxpy as cp
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.coeff_extractor import CoeffExtractor

x = cp.Variable((2,))
x.value = [1.0, 2.0]
y = cp.Variable()
y.value = 4.0

expr = cp.sum(x) + y + 5 + 2*y + x[0] * 3
constr = [x + y <= 5]

problem = cp.Problem(cp.Minimize(expr), constr)
inverse_data = InverseData(problem)
extractor = CoeffExtractor(inverse_data, cp.SCIPY_CANON_BACKEND)
print(str(constr[0].expr))
A = extractor.affine([expr, constr[0].expr])
print(A.toarray().reshape((3,4), order='F'))
