#ifndef PROBLEM_UPDATE_PARAMS_H
#define PROBLEM_UPDATE_PARAMS_H

#include "common.h"

static PyObject *py_problem_update_params(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *theta_obj;

    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &theta_obj))
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

    PyArrayObject *theta_array = (PyArrayObject *) PyArray_FROM_OTF(
        theta_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!theta_array)
    {
        return NULL;
    }

    int theta_size = (int) PyArray_SIZE(theta_array);
    int expected = 0;
    for (int i = 0; i < prob->n_param_nodes; i++)
        expected += prob->param_nodes[i]->size;
    if (theta_size != expected)
    {
        Py_DECREF(theta_array);
        PyErr_Format(PyExc_ValueError,
                     "theta size %d does not match expected %d",
                     theta_size, expected);
        return NULL;
    }

    problem_update_params(prob, (const double *) PyArray_DATA(theta_array));
    Py_DECREF(theta_array);

    Py_RETURN_NONE;
}

#endif /* PROBLEM_UPDATE_PARAMS_H */
