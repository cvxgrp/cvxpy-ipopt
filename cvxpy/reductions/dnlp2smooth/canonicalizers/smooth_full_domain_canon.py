"""
Copyright 2025 CVXPY developers

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
from cvxpy.expressions.variable import Variable


def smooth_full_domain_canon_non_chain_rule(expr, args):
    if isinstance(args[0], Variable):
        return expr.copy([args[0]]), []
    t = Variable(args[0].shape)
    if args[0].value is not None:
        t.value = args[0].value
    return expr.copy([t]), [t == args[0]]

def smooth_full_domain_canon_chain_rule(expr, args):
    return expr.copy(args), []

# prod does not have chain rule implemented in diff engine
prod_canon = smooth_full_domain_canon_non_chain_rule

# quad_form has chain rule implemented in diff engine
quad_form_canon = smooth_full_domain_canon_chain_rule

# these have chain rule implemented in diff engine
exp_canon = smooth_full_domain_canon_chain_rule
sin_canon = smooth_full_domain_canon_chain_rule
cos_canon = smooth_full_domain_canon_chain_rule
sinh_canon = smooth_full_domain_canon_chain_rule
tanh_canon = smooth_full_domain_canon_chain_rule
asinh_canon = smooth_full_domain_canon_chain_rule
logistic_canon = smooth_full_domain_canon_chain_rule
normcdf_canon = smooth_full_domain_canon_chain_rule

# bivariate atoms with chain rule implemented in diff engine
multiply_canon = smooth_full_domain_canon_chain_rule
matmul_canon = smooth_full_domain_canon_chain_rule


# TODO: do we even need the smooth full domain canon chain rule canonicalizers?
