#ifndef ATOM_PARAM_SCALAR_MULT_H
#define ATOM_PARAM_SCALAR_MULT_H

#include "bivariate.h"
#include "common.h"

/* Parameter scalar multiplication: p * f(x) where p is a parameter node */
static PyObject *py_make_param_scalar_mult(PyObject *self, PyObject *args)
{
    PyObject *param_capsule;
    PyObject *child_capsule;

    if (!PyArg_ParseTuple(args, "OO", &param_capsule, &child_capsule))
    {
        return NULL;
    }

    expr *param_node =
        (expr *) PyCapsule_GetPointer(param_capsule, EXPR_CAPSULE_NAME);
    if (!param_node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid parameter capsule");
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_param_scalar_mult(param_node, child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create param_scalar_mult node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_PARAM_SCALAR_MULT_H */
