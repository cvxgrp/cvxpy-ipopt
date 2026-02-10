#ifndef PROBLEM_REGISTER_PARAMS_H
#define PROBLEM_REGISTER_PARAMS_H

#include "common.h"

static PyObject *py_problem_register_params(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *param_list;

    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &param_list))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    if (!PyList_Check(param_list))
    {
        PyErr_SetString(PyExc_TypeError, "param_list must be a list");
        return NULL;
    }

    Py_ssize_t n_param_nodes = PyList_Size(param_list);
    if (n_param_nodes == 0)
    {
        Py_RETURN_NONE;
    }

    expr **param_nodes = (expr **) malloc(n_param_nodes * sizeof(expr *));
    if (!param_nodes)
    {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n_param_nodes; i++)
    {
        PyObject *capsule = PyList_GetItem(param_list, i);
        param_nodes[i] =
            (expr *) PyCapsule_GetPointer(capsule, EXPR_CAPSULE_NAME);
        if (!param_nodes[i])
        {
            free(param_nodes);
            PyErr_SetString(PyExc_ValueError, "invalid parameter capsule in list");
            return NULL;
        }
    }

    problem_register_params(prob, param_nodes, (int) n_param_nodes);
    free(param_nodes);

    Py_RETURN_NONE;
}

#endif /* PROBLEM_REGISTER_PARAMS_H */
