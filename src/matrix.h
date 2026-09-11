#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

typedef struct {
    int num_layers;
    Matrix *weights;
    Matrix *biases;
} Network;

Matrix CreateMatrix(int rows, int cols);
void FreeMatrix(Matrix *m);
void PrintMatrix(Matrix m);
Matrix MultiplyMatrix(Matrix a, Matrix b);
Matrix AddMatrix(Matrix a, Matrix b);
Matrix TransposeMatrix(Matrix m);
Matrix ElementwiseMultiplyMatrix(Matrix a, Matrix b);
Matrix ScalarMultiplyMatrix(Matrix m, double scalar);
Matrix SubstractMatrix(Matrix a, Matrix b);
Matrix Sigmoid(Matrix m);
Matrix SigmoidDerivative(Matrix m);
Matrix ForwardPass(Matrix X, Matrix W1, Matrix b1, Matrix W2, Matrix b2);
double TrainStep(Matrix X, Matrix y, Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2, double learning_rate);
double RandWeight();

Network CreateNetwork(int num_sizes, int *layer_sizes);
void FreeNetwork(Network *network);
Matrix NetworkForward(Network network, Matrix input);
double NetworkTrainStep(Network *network, Matrix X, Matrix y, double learning_rate);

#endif
