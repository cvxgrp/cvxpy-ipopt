import numpy as np
import torch

import cvxpy as cp
from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.reductions.utilities import (
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)
from cvxtorch import TorchExpression


class HS071():
    def __init__(self, problem: cp.Problem):
        self.problem = problem
        self.main_var = []
        for var in self.problem.variables():
            self.main_var.append(var)
    
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        # Set the variable value
        offset = 0
        for var in self.main_var:
            size = var.size
            var.value = x[offset:offset+size]
            offset += size
        # Evaluate the objective
        obj_value = self.problem.objective.args[0].value
        return obj_value
    
    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        # Convert to torch tensor with gradient tracking
        offset = 0
        torch_exprs = []
        for var in self.main_var:
            size = var.size
            slice = x[offset:offset+size]
            torch_exprs.append(torch.from_numpy(slice.astype(np.float64)).requires_grad_(True))
            offset += size
        
        torch_obj = TorchExpression(self.problem.objective.args[0]).torch_expression(*torch_exprs)
        
        # Compute gradient
        torch_obj.backward()
        gradients = []
        for tensor in torch_exprs:
            if tensor.grad is not None:
                gradients.append(tensor.grad.detach().numpy().flatten())
            else:
                gradients.append(np.array([0] * tensor.numel()))
        return np.concatenate(gradients)
    
    def constraints(self, x):
        """Returns the constraint values."""
        # Set the variable value
        offset = 0
        for var in self.main_var:
            size = var.size
            var.value = x[offset:offset+size]
            offset += size
        
        # Evaluate all constraints
        constraint_values = []
        for constraint in self.problem.constraints:
            constraint_values.append(np.asarray(constraint.args[0].value).flatten())
        return np.concat(constraint_values)
    
    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        # Convert to torch tensor with gradient tracking
        offset = 0
        torch_vars_dict = {}
        torch_exprs = []
        
        for var in self.main_var:
            size = var.size
            slice = x[offset:offset+size]
            torch_tensor = torch.from_numpy(slice.astype(np.float64)).requires_grad_(True)
            torch_vars_dict[var.id] = torch_tensor  # Map CVXPY variable ID to torch tensor
            torch_exprs.append(torch_tensor)
            offset += size
        
        # Define a function that computes all constraint values
        def constraint_function(*args):
            # Create mapping from torch tensors back to CVXPY variables
            torch_to_var = {}
            for i, var in enumerate(self.main_var):
                torch_to_var[var.id] = args[i]
            
            constraint_values = []
            for constraint in self.problem.constraints:
                constraint_expr = constraint.args[0]
                constraint_vars = constraint_expr.variables()
                
                # Create ordered list of torch tensors for this specific constraint
                # in the order that the constraint expression expects them
                constr_torch_args = []
                for var in constraint_vars:
                    if var.id in torch_to_var:
                        constr_torch_args.append(torch_to_var[var.id])
                    else:
                        raise ValueError(f"Variable {var} not found in torch mapping")
                
                torch_expr = TorchExpression(constraint_expr).torch_expression(*constr_torch_args)
                constraint_values.append(torch_expr)
            return torch.cat([torch.atleast_1d(cv) for cv in constraint_values])

        # Compute Jacobian using torch.autograd.functional.jacobian
        if len(self.problem.constraints) > 0:
            jacobian_tuple = torch.autograd.functional.jacobian(constraint_function, 
                                                                tuple(torch_exprs))
            # Handle the case where jacobian_tuple is a tuple (multiple variables)
            if isinstance(jacobian_tuple, tuple):
                # Concatenate along the last dimension (variable dimension)
                jacobian_matrix = torch.cat(jacobian_tuple, dim=-1)
            else:
                # Single variable case
                jacobian_matrix = jacobian_tuple
            return jacobian_matrix.detach().numpy()

class Bounds_Getter():
    def __init__(self, problem: cp.Problem):
        self.problem = problem
        self.main_var = problem.variables()
        self.get_constraint_bounds()
        self.get_variable_bounds()

    def get_constraint_bounds(self):
        """Also normalizes the constraints and creates a new problem"""
        lower = []
        upper = []
        new_constr = []
        
        for constraint in self.problem.constraints:
            if isinstance(constraint, Equality):
                lower.extend([0.0] * constraint.size)
                upper.extend([0.0] * constraint.size)
                new_constr.append(lower_equality(constraint))
            elif isinstance(constraint, Inequality):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(lower_ineq_to_nonneg(constraint))
            elif isinstance(constraint, NonPos):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(nonpos2nonneg(constraint))
        
        lowered_con_problem = self.problem.copy([self.problem.objective, new_constr])
        self.new_problem = lowered_con_problem
        self.cl = np.array(lower)
        self.cu = np.array(upper)

    def get_variable_bounds(self):
        var_lower = []
        var_upper = []
        for var in self.main_var:
            size = var.size
            if var.bounds:
                var_lower.extend(var.bounds[0])
                var_upper.extend(var.bounds[1])
            else:
                # No bounds specified, use infinite bounds
                var_lower.extend([-np.inf] * size)
                var_upper.extend([np.inf] * size)

        self.lb = np.array(var_lower)
        self.ub = np.array(var_upper)
