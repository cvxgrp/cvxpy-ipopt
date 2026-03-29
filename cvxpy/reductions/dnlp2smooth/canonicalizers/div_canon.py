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

import numpy as np

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.warn import warn

MIN_INIT = 1e-3

# We canonicalize div(f(x), g(x)) as f(x) * power(z, -1), z = g(x), z >= 0.
def div_canon(expr, args):
    
    if not args[1].is_nonneg():
        warn(
            "CVXPY (DNLP) could not verify that the denominator of a division "
            "appearing in your problem is nonnegative. The solver will proceed "
            "under the assumption that the denominator is nonnegative. If this "
            "assumption is incorrect, the solution may be invalid. "
        )
    
    z = Variable(args[1].shape, nonneg=True)

    if args[1].value is not None:
        z.value = np.maximum(args[1].value, MIN_INIT)

    # TODO (dance858): should. multiply and power be other canons?
    return multiply(args[0], power(z, -1)), [z == args[1]]


# ----------------------------------------------------------------------------------
#                                Old version
# ----------------------------------------------------------------------------------
# We canonicalize div(f(x), g(x)) as z * y = f(x), y = g(x), y >= 0.
# In other words, it assumes that the denominator is nonnegative.
#def div_canon_old(expr, args):
#
#    # raise an error if the denominator is not nonnegative
#    if not args[1].is_nonneg():
#        raise ValueError("The denominator of a division must be nonnegative. "
#                          "Did you forget to specify bounds?")
#
#    dim = args[0].shape
#    sgn_z = args[0].sign
#
#    if sgn_z == 'NONNEGATIVE':
#        z = Variable(dim, nonneg=True)
#    elif sgn_z == 'NONPOSITIVE':
#        z = Variable(dim, nonpos=True)
#    else:
#        z = Variable(dim)
#
#    y = Variable(args[1].shape, nonneg=True)
#
#    if args[0].value is not None and args[1].value is not None:
#        y.value = np.maximum(args[1].value, MIN_INIT)
#        val = args[0].value / y.value
#
#        # dimension hack
#        if dim == () and val.shape == (1,):
#            z.value = val[0]
#        else:
#            z.value = val
#
#    return z, [multiply(z, y) == args[0], y == args[1]]
