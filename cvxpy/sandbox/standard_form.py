import cvxpy as cp

n = 5
x = cp.Variable(n)
# form a simple linear programming problem
objective = cp.Minimize(cp.sum(x))
constraints = [x >= 0, cp.sum(x) == 1]
problem = cp.Problem(objective, constraints)
data, chain, inverse_data = problem.get_problem_data(cp.CLARABEL)
print(data)
print(data['A'].toarray())
print(data['b'])
print("----- Chain -----")
print(chain)
print("----- Inverse Data -----")
print(inverse_data)