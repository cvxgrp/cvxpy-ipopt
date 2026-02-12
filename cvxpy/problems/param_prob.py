"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import abc

import numpy as np


class ParamProb(metaclass=abc.ABCMeta):
    """An abstract base class for parameterized problems.

    Parameterized problems are produced during the first canonicalization
    and allow canonicalization to be short-circuited for future solves.
    """

    @abc.abstractmethod
    def is_mixed_integer(self) -> bool:
        """Is the problem mixed-integer?"""
        raise NotImplementedError()

    @abc.abstractmethod
    def apply_parameters(self, id_to_param_value=None, zero_offset: bool = False,
                         keep_zeros: bool = False):
        """Returns A, b after applying parameters (and reshaping).

        Args:
          id_to_param_value: (optional) dict mapping parameter ids to values
          zero_offset: (optional) if True, zero out the constant offset in the
                       parameter vector
          keep_zeros: (optional) if True, store explicit zeros in A where
                        parameters are affected
        """
        raise NotImplementedError()

    def split_solution(self, sltn, active_vars=None):
        """Splits the solution into individual variables."""
        # Import here to avoid circular imports.
        from cvxpy.reductions import cvx_attr2constr
        if active_vars is None:
            active_vars = [v.id for v in self.variables]
        sltn_dict = {}
        for var_id, col in self.var_id_to_col.items():
            if var_id in active_vars:
                var = self.id_to_var[var_id]
                value = sltn[col:var.size + col]
                if var.attributes_were_lowered():
                    orig_var = var.variable_of_provenance()
                    value = cvx_attr2constr.recover_value_for_variable(
                        orig_var, value, project=False)
                    sltn_dict[orig_var.id] = np.reshape(
                        value, orig_var.shape, order='F')
                else:
                    sltn_dict[var_id] = np.reshape(
                        value, var.shape, order='F')
        return sltn_dict
