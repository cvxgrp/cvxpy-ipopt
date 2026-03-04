"""Centralized import for the sparsediffpy C bindings.

Copyright 2025, the CVXPY developers

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

try:
    from sparsediffpy import _sparsediffengine as _diffengine
except ImportError as e:
    raise ImportError(
        "NLP support requires sparsediffpy. Install with: pip install sparsediffpy"
    ) from e

__all__ = ["_diffengine"]
