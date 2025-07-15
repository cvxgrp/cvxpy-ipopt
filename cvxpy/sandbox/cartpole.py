import cvxpy as cp
import numpy as np

# taken from https://github.com/kevin-tracy/lazy_nlp_qd.jl/blob/main/test/trajopt_test.jl
# and modified to use CVXPY with Claude
def create_idx(nx, nu, N):
    """Create indexing tools for decision variable vector Z."""
    nz = (N-1) * nu + N * nx
    x = [np.arange((i-1)*(nx+nu), (i-1)*(nx+nu)+nx) for i in range(1, N+1)]
    u = [np.arange((i-1)*(nx+nu)+nx, (i-1)*(nx+nu)+nx+nu) for i in range(1, N)]
    c = [np.arange((i-1)*nx, i*nx) for i in range(1, N)]
    nc = (N-1) * nx
    
    return {
        'nx': nx, 'nu': nu, 'N': N, 'nz': nz, 'nc': nc,
        'x': x, 'u': u, 'c': c
    }

# Problem size
nx = 4
nu = 1
dt = 0.05
tf = 2.0
t_vec = np.arange(0, tf + dt, dt)
N = len(t_vec)

# LQR cost matrices
Q = np.eye(nx)
R = 0.1 * np.eye(nu)
Qf = 10 * np.eye(nx)

# Indexing
idx = create_idx(nx, nu, N)

# Initial and goal states
x0 = np.array([0, 0, 0, 0])
xf = np.array([0, np.pi, 0, 0])
Xref = [xf for _ in range(N)]
Uref = [np.zeros(nu) for _ in range(N-1)]

# System parameters
params = {
    'Q': Q, 'R': R, 'Qf': Qf,
    'x0': x0, 'xf': xf,
    'Xref': Xref, 'Uref': Uref,
    'dt': dt, 'N': N, 'idx': idx,
    'mc': 1.0, 'mp': 0.2, 'l': 0.5
}

# Create decision variables
Z = cp.Variable(idx['nz'])

# Extract state and control variables
X = [Z[idx['x'][i]] for i in range(N)]
U = [Z[idx['u'][i]] for i in range(N-1)]

# Define cost function
cost = 0
for i in range(N-1):
    dx = X[i] - Xref[i]
    du = U[i] - Uref[i]
    cost += 0.5 * cp.quad_form(dx, Q) + 0.5 * cp.quad_form(du, R)

dx = X[N-1] - Xref[N-1]
cost += 0.5 * cp.quad_form(dx, Qf)

# Define constraints
constraints = []

# Initial condition
constraints.append(X[0] == x0)

# Terminal condition
constraints.append(X[N-1] == xf)

# Dynamics constraints using RK4
mc = params['mc']
mp = params['mp']
l = params['l']
g = 9.81

for i in range(N-1):
    xi = X[i]
    ui = U[i]
    
    # Define RK4 integration for cartpole dynamics
    def dynamics_step(x, u):
        q = x[:2]
        qd = x[2:4]
        
        s = cp.sin(q[1])
        c = cp.cos(q[1])
        
        # Mass matrix and forces
        H11 = mc + mp
        H12 = mp * l * c
        H21 = mp * l * c
        H22 = mp * l * l
        
        C1 = -mp * qd[1] * l * s * qd[1]
        C2 = 0
        
        G1 = 0
        G2 = mp * g * l * s
        
        B1 = 1
        B2 = 0
        
        # H * qdd + C * qd + G = B * u
        # qdd = H^(-1) * (B * u - C * qd - G)
        det_H = H11 * H22 - H12 * H21
        
        rhs1 = B1 * u[0] - C1 - G1
        rhs2 = B2 * u[0] - C2 - G2
        
        qdd1 = (H22 * rhs1 - H12 * rhs2) / det_H
        qdd2 = (-H21 * rhs1 + H11 * rhs2) / det_H
        
        return cp.hstack([qd[0], qd[1], qdd1, qdd2])
    
    # RK4 steps
    k1 = dt * dynamics_step(xi, ui)
    k2 = dt * dynamics_step(xi + k1/2, ui)
    k3 = dt * dynamics_step(xi + k2/2, ui)
    k4 = dt * dynamics_step(xi + k3, ui)
    
    xi_next = xi + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Dynamics constraint
    constraints.append(X[i+1] == xi_next)

# Control bounds
u_min = -40 * np.ones(nu)
u_max = 40 * np.ones(nu)
for i in range(N-1):
    constraints.append(U[i] >= u_min)
    constraints.append(U[i] <= u_max)

# State bounds (excluding initial and final states)
x_min = -150 * np.ones(nx)
x_max = 150 * np.ones(nx)
for i in range(1, N-1):
    constraints.append(X[i] >= x_min)
    constraints.append(X[i] <= x_max)

# Create and solve the problem
problem = cp.Problem(cp.Minimize(cost), constraints)

# Initial guess
Z0 = 0.01 * np.random.randn(idx['nz'])

# Solve with tolerances similar to the Julia code
problem.solve(solver=cp.IPOPT, 
              verbose=True, nlp=True)

# Extract solution
X_sol = [X[i].value for i in range(N)]
U_sol = [U[i].value for i in range(N-1)]

print(f"\nOptimal cost: {problem.value}")
print(f"Solver status: {problem.status}")
print(f"\nFinal state: {X_sol[-1]}")
print(f"Initial control: {U_sol[0]}")

# You can plot the results using:
import matplotlib.pyplot as plt

X_array = np.array(X_sol).T
plt.plot(t_vec, X_array[0, :], label='x position')
plt.plot(t_vec, X_array[1, :], label='theta')
plt.plot(t_vec, X_array[2, :], label='x velocity')
plt.plot(t_vec, X_array[3, :], label='theta velocity')
plt.legend()
plt.show()