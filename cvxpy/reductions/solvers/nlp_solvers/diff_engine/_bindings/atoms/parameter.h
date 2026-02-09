#ifndef ATOM_PARAMETER_H
#define ATOM_PARAMETER_H

#include "common.h"

static PyObject *py_make_parameter(PyObject *self, PyObject *args)
{
    int d1, d2, param_id, n_vars;
    if (!PyArg_ParseTuple(args, "iiii", &d1, &d2, &param_id, &n_vars))
    {
        return NULL;
    }

    expr *node = new_parameter(d1, d2, param_id, n_vars);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create parameter node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_PARAMETER_H */
