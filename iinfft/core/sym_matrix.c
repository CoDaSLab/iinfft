#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <complex.h>

/* Function to calculate the full symmetric matrix (transpose) */
static PyObject *compute_symmetric_matrix(PyObject *self, PyObject *args) {
    PyObject *f_j_obj, *h_k_obj;  // Input arrays for f_j and h_k

    // Parse Python arguments: two NumPy arrays (f_j and h_k)
    if (!PyArg_ParseTuple(args, "OO", &f_j_obj, &h_k_obj)) {
        return NULL;
    }

    // Convert to NumPy arrays
    PyArrayObject *f_j_array = (PyArrayObject *)PyArray_FROM_OTF(f_j_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *h_k_array = (PyArrayObject *)PyArray_FROM_OTF(h_k_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!f_j_array || !h_k_array) {
        Py_XDECREF(f_j_array);
        Py_XDECREF(h_k_array);
        return NULL;
    }

    // Get dimensions
    int M = (int)PyArray_SIZE(f_j_array);  // Length of f_j
    int N = (int)PyArray_SIZE(h_k_array);  // Length of h_k

    // Create the output symmetric matrix (complex type)
    npy_intp dims[2] = {N, N};
    PyArrayObject *sym_matrix = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_COMPLEX128, 0);

    // Calculate the transposed matrix
    double *f_j = (double *)PyArray_DATA(f_j_array);
    double *h_k = (double *)PyArray_DATA(h_k_array);
    npy_complex128 *matrix_data = (npy_complex128 *)PyArray_DATA(sym_matrix);

    for (int k1 = 0; k1 < N; ++k1) {
        for (int k2 = 0; k2 < N; ++k2) {  // Iterate over the entire matrix
            double sum_re = 0.0;  // Real part
            double sum_im = 0.0;  // Imaginary part
            double delta_h = h_k[k1] - h_k[k2];

            for (int j = 0; j < M; ++j) {
                double phase = 2.0 * M_PI * f_j[j] * delta_h;
                double cos_phase = cos(phase);
                double sin_phase = sin(phase);

                sum_re += cos_phase;
                sum_im += sin_phase;
            }

            // Store the result in the matrix (transpose: swap k1 and k2)
            npy_complex128 value = sum_re + sum_im * I;  // Construct complex number
            matrix_data[k2 * N + k1] = value;  // Swap k1 and k2
        }
    }

    // Cleanup
    Py_DECREF(f_j_array);
    Py_DECREF(h_k_array);

    return (PyObject *)sym_matrix;
}

/* Module method table */
static PyMethodDef SymMatrixMethods[] = {
    {"compute_symmetric_matrix", compute_symmetric_matrix, METH_VARARGS, "Compute full symmetric matrix (transposed) from irregular time samples and regular frequencies."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symmatrixmodule = {
    PyModuleDef_HEAD_INIT,
    "sym_matrix",
    "Compute full symmetric matrix (transposed) using NumPy C API.",
    -1,
    SymMatrixMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_sym_matrix(void) {
    import_array();  // Necessary for NumPy
    return PyModule_Create(&symmatrixmodule);
}
