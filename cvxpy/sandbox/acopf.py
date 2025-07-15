import cvxpy as cp
import numpy as np

N = 4

# Conductance/susceptance components
G = np.array(
    [
        [1.7647, -0.5882, 0.0, -1.1765],
        [-0.5882, 1.5611, -0.3846, -0.5882],
        [0.0, -0.3846, 1.5611, -1.1765],
        [-1.1765, -0.5882, -1.1765, 2.9412],
    ]
)

B = np.array(
    [
        [-7.0588, 2.3529, 0.0, 4.7059],
        [2.3529, -6.629, 1.9231, 2.3529],
        [0.0, 1.9231, -6.629, 4.7059],
        [4.7059, 2.3529, 4.7059, -11.7647],
    ]
)

# Assign bounds where fixings are needed
v_lb = np.array([1.0, 0.0, 1.0, 0.0])
v_ub = np.array([1.0, 1.5, 1.0, 1.5])

P_lb = np.array([-3.0, -0.3, 0.3, -0.2])
P_ub = np.array([3.0, -0.3, 0.3, -0.2])

Q_lb = np.array([-3.0, -0.2, -3.0, -0.15])
Q_ub = np.array([3.0, -0.2, 3.0, -0.15])

theta_lb = np.array([0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2])
theta_ub = np.array([0.0, np.pi / 2, np.pi / 2, np.pi / 2])

# Create variables with bounds
P = cp.Variable(N, name="P")
P.value = np.random.uniform(P_lb, P_ub)  # Real power for buses
Q = cp.Variable(N, name="Q")
Q.value = np.random.uniform(Q_lb, Q_ub)  # Reactive power for buses
v = cp.Variable(N, name="v")
v.value = np.random.uniform(v_lb, v_ub)  # Voltage magnitude at buses
theta = cp.Variable(N, name="theta")
theta.value = np.random.uniform(theta_lb, theta_ub)  # Voltage angle at buses

# Reshape theta to column vector for broadcasting
theta_col = cp.reshape(theta, (N, 1))

# Create constraints list
constraints = []

# Add bound constraints
constraints += [
    P >= P_lb,
    P <= P_ub,
    Q >= Q_lb,
    Q <= Q_ub,
    v >= v_lb,
    v <= v_ub,
    theta >= theta_lb,
    theta <= theta_ub
]
P_balance = cp.multiply(v, (G * cp.cos(theta_col - theta_col.T) + B * cp.sin(theta_col - theta_col.T)) @ v)
constraints.append(P == P_balance)

# Reactive power balance
Q_balance = cp.multiply(v, (G * cp.sin(theta_col - theta_col.T) - B * cp.cos(theta_col - theta_col.T)) @ v)
constraints.append(Q == Q_balance)

# Objective: minimize reactive power at buses 1 and 3 (indices 0 and 2)
objective = cp.Minimize(Q[0] + Q[2])

# Create and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, nlp=True)
