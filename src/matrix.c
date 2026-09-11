#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

Matrix CreateMatrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = calloc(rows * cols, sizeof(double));

    if (m.data == NULL) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }

    return m;
}

void FreeMatrix(Matrix *m) {
    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

void PrintMatrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            printf("%f ", m.data[i * m.cols + j]);
        }
        printf("\n");
    }
}

Matrix MultiplyMatrix(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
        fprintf(stderr, "dimension mismatch (%dx%d) * (%dx%d)\n", a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }

    Matrix result = CreateMatrix(a.rows, b.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            double sum = 0;
            for (int k = 0; k < a.cols; k++) {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }

    return result;
}

Matrix AddMatrix(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "dimension mismatch (%dx%d) * (%dx%d)\n", a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }

    Matrix result = CreateMatrix(a.rows, a.cols);
    
    for (int i = 0; i < a.rows * a.cols; i++) {
        result.data[i] = a.data[i] + b.data[i];
    }

    return result;
}

Matrix TransposeMatrix(Matrix m) {
    Matrix result = CreateMatrix(m.cols, m.rows);

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.data[j * result.cols + i] = m.data[i * m.cols + j];
        }
    }

    return result;
}

Matrix ElementwiseMultiplyMatrix(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "dimension mismatch (%dx%d) * (%dx%d)\n", a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }

    Matrix result = CreateMatrix(a.rows, a.cols);

    for (int i = 0; i < a.rows * a.cols; i++) {
        result.data[i] = a.data[i] * b.data[i];
    }

    return result;
}

Matrix ScalarMultiplyMatrix(Matrix m, double scalar) {
    Matrix result = CreateMatrix(m.rows, m.cols);

    for (int i = 0; i < m.rows * m.cols; i++) {
        result.data[i] = m.data[i] * scalar;
    }

    return result;
}

Matrix SubstractMatrix(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "dimension mismatch (%dx%d) * (%dx%d)\n", a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }

    Matrix result = CreateMatrix(a.rows, a.cols);

    for (int i = 0; i < a.rows * a.cols; i++) {
        result.data[i] = a.data[i] - b.data[i];
    }

    return result;
}

