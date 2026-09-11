#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

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

Matrix Sigmoid(Matrix m) {
    Matrix result = CreateMatrix(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.data[i * m.cols + j] = 1 / (1 + exp(-m.data[i * m.cols + j]));
        }
    }

    return result;
}

Matrix SigmoidDerivative(Matrix m) {
    Matrix result = CreateMatrix(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.data[i * m.cols + j] = m.data[i * m.cols + j] * (1 - m.data[i * m.cols + j]);
        }
    }

    return result;
}

Matrix ForwardPass(Matrix X, Matrix W1, Matrix b1, Matrix W2, Matrix b2) {
    Matrix z1 = MultiplyMatrix(W1, X);
    Matrix a1 = AddMatrix(z1, b1);
    FreeMatrix(&z1);
    Matrix a1_sigmoid = Sigmoid(a1);

    Matrix z2 = MultiplyMatrix(W2, a1_sigmoid);
    Matrix a2 = AddMatrix(z2, b2);
    FreeMatrix(&z2);
    Matrix a2_sigmoid = Sigmoid(a2);

    return a2_sigmoid;
}

double TrainStep(Matrix X, Matrix y, Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2, double learning_rate) {
    Matrix z1 = MultiplyMatrix(*W1, X);
    Matrix a1 = AddMatrix(z1, *b1);
    FreeMatrix(&z1);
    Matrix a1_sigmoid = Sigmoid(a1);
    FreeMatrix(&a1);

    Matrix z2 = MultiplyMatrix(*W2, a1_sigmoid);
    Matrix a2 = AddMatrix(z2, *b2);
    FreeMatrix(&z2);
    Matrix predictions = Sigmoid(a2);
    FreeMatrix(&a2);

    double loss = 0;
    for (int i = 0; i < predictions.rows * predictions.cols; i++) {
        double diff = predictions.data[i] - y.data[i];
        loss += diff * diff;
    }

    Matrix error = SubstractMatrix(predictions, y);
    Matrix d_sig2 = SigmoidDerivative(predictions);
    Matrix delta2 = ElementwiseMultiplyMatrix(error, d_sig2);
    FreeMatrix(&error);
    FreeMatrix(&d_sig2);
    FreeMatrix(&predictions);

    Matrix transposed_a1 = TransposeMatrix(a1_sigmoid);
    Matrix grad_W2 = MultiplyMatrix(delta2, transposed_a1);
    FreeMatrix(&transposed_a1);

    Matrix transposed_W2 = TransposeMatrix(*W2);
    Matrix backprop2 = MultiplyMatrix(transposed_W2, delta2);
    FreeMatrix(&transposed_W2);

    Matrix d_sig1 = SigmoidDerivative(a1_sigmoid);
    Matrix delta1 = ElementwiseMultiplyMatrix(backprop2, d_sig1);
    FreeMatrix(&backprop2);
    FreeMatrix(&d_sig1);
    FreeMatrix(&a1_sigmoid);

    Matrix transposed_X = TransposeMatrix(X);
    Matrix grad_W1 = MultiplyMatrix(delta1, transposed_X);
    FreeMatrix(&transposed_X);

    Matrix scaled_grad_W2 = ScalarMultiplyMatrix(grad_W2, learning_rate);
    Matrix new_W2 = SubstractMatrix(*W2, scaled_grad_W2);
    FreeMatrix(&grad_W2);
    FreeMatrix(&scaled_grad_W2);
    FreeMatrix(W2);
    *W2 = new_W2;

    Matrix scaled_grad_b2 = ScalarMultiplyMatrix(delta2, learning_rate);
    Matrix new_b2 = SubstractMatrix(*b2, scaled_grad_b2);
    FreeMatrix(&scaled_grad_b2);
    FreeMatrix(b2);
    *b2 = new_b2;

    Matrix scaled_grad_W1 = ScalarMultiplyMatrix(grad_W1, learning_rate);
    Matrix new_W1 = SubstractMatrix(*W1, scaled_grad_W1);
    FreeMatrix(&grad_W1);
    FreeMatrix(&scaled_grad_W1);
    FreeMatrix(W1);
    *W1 = new_W1;

    Matrix scaled_grad_b1 = ScalarMultiplyMatrix(delta1, learning_rate);
    Matrix new_b1 = SubstractMatrix(*b1, scaled_grad_b1);
    FreeMatrix(&scaled_grad_b1);
    FreeMatrix(b1);
    *b1 = new_b1;

    FreeMatrix(&delta1);
    FreeMatrix(&delta2);

    return loss;
}

double RandWeight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

Network CreateNetwork(int num_sizes, int *layer_sizes) {
    Network network;
    network.num_layers = num_sizes - 1;
    network.weights = malloc(network.num_layers * sizeof(Matrix));
    network.biases = malloc(network.num_layers * sizeof(Matrix));

    for (int i = 0; i < network.num_layers; i++) {
        network.weights[i] = CreateMatrix(layer_sizes[i + 1], layer_sizes[i]);
        network.biases[i] = CreateMatrix(layer_sizes[i + 1], 1);

        for (int j = 0; j < network.weights[i].rows * network.weights[i].cols; j++) {
            network.weights[i].data[j] = RandWeight();
        }

        for (int j = 0; j < network.weights[i].rows; j++) {
            network.biases[i].data[j] = RandWeight();
        }
    }

    return network;
}

void FreeNetwork(Network *network) {
    for (int i = 0; i < network->num_layers; i++) {
        FreeMatrix(&network->weights[i]);
        FreeMatrix(&network->biases[i]);
    }
    free(network->weights);
    free(network->biases);
    network->weights = NULL;
    network->biases = NULL;
    network->num_layers = 0;
}

Matrix NetworkForward(Network network, Matrix input) {
    Matrix current = input;
    int owns_current = 0;

    for (int i = 0; i < network.num_layers; i++) {
        Matrix z = MultiplyMatrix(network.weights[i], current);
        Matrix a = AddMatrix(z, network.biases[i]);
        Matrix a_sigmoid = Sigmoid(a);

        FreeMatrix(&z);
        FreeMatrix(&a);

        if (owns_current) {
            FreeMatrix(&current);
        }
        current = a_sigmoid;
        owns_current = 1;
    }

    return current;
}

double NetworkTrainStep(Network *network, Matrix X, Matrix y, double learning_rate) {
    Matrix *activations = malloc((network->num_layers + 1) * sizeof(Matrix));
    activations[0] = X;

    for (int i = 0; i < network->num_layers; i++) {
        Matrix z = MultiplyMatrix(network->weights[i], activations[i]);
        Matrix a = AddMatrix(z, network->biases[i]);
        Matrix a_sigmoid = Sigmoid(a);

        FreeMatrix(&z);
        FreeMatrix(&a);

        activations[i + 1] = a_sigmoid;
    }

    Matrix output = activations[network->num_layers];

    double loss = 0;
    for (int i = 0; i < output.rows * output.cols; i++) {
        double diff = output.data[i] - y.data[i];
        loss += diff * diff;
    }

    Matrix error = SubstractMatrix(output, y);
    Matrix d_sig = SigmoidDerivative(output);
    Matrix delta = ElementwiseMultiplyMatrix(error, d_sig);
    FreeMatrix(&error);
    FreeMatrix(&d_sig);

    for (int layer = network->num_layers - 1; layer >= 0; layer--) {
        Matrix prev_activation_T = TransposeMatrix(activations[layer]);
        Matrix grad_W = MultiplyMatrix(delta, prev_activation_T);
        FreeMatrix(&prev_activation_T);

        Matrix next_delta;
        int has_next_delta = 0;

        if (layer > 0) {
            Matrix W_T = TransposeMatrix(network->weights[layer]);
            Matrix backprop = MultiplyMatrix(W_T, delta);
            FreeMatrix(&W_T);

            Matrix d_sig_prev = SigmoidDerivative(activations[layer]);
            next_delta = ElementwiseMultiplyMatrix(backprop, d_sig_prev);
            FreeMatrix(&backprop);
            FreeMatrix(&d_sig_prev);
            has_next_delta = 1;
        }

        Matrix scaled_grad_W = ScalarMultiplyMatrix(grad_W, learning_rate);
        Matrix new_W = SubstractMatrix(network->weights[layer], scaled_grad_W);
        FreeMatrix(&grad_W);
        FreeMatrix(&scaled_grad_W);
        FreeMatrix(&network->weights[layer]);
        network->weights[layer] = new_W;

        Matrix scaled_delta = ScalarMultiplyMatrix(delta, learning_rate);
        Matrix new_b = SubstractMatrix(network->biases[layer], scaled_delta);
        FreeMatrix(&scaled_delta);
        FreeMatrix(&network->biases[layer]);
        network->biases[layer] = new_b;

        FreeMatrix(&delta);
        if (has_next_delta) {
            delta = next_delta;
        }
    }

    for (int i = 1; i <= network->num_layers; i++) {
        FreeMatrix(&activations[i]);
    }
    free(activations);

    return loss;
}
