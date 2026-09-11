#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "mnist.h"

static int argmax(Matrix m) {
    int best = 0;
    for (int i = 1; i < m.rows; i++) {
        if (m.data[i] > m.data[best]) best = i;
    }
    return best;
}

static double evaluate(Network network, Dataset d) {
    int correct = 0;
    for (int i = 0; i < d.num; i++) {
        Matrix output = NetworkForward(network, d.images[i]);
        if (argmax(output) == argmax(d.labels[i])) {
            correct++;
        }
        FreeMatrix(&output);
    }

    return (double) correct / d.num;
}

int main() {
    srand(42);

    printf("loading MNIST...\n");
    Dataset train = LoadMnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    Dataset test = LoadMnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    printf("train size: %d, test size: %d\n", train.num, test.num);
    
    int layer_sizes[] = {784, 128, 10};
    Network network = CreateNetwork(3, layer_sizes);
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < 10; epoch++) {
        double total_loss = 0;

        for (int i = 0; i < train.num; i++) {
            total_loss += NetworkTrainStep(&network, train.images[i], train.labels[i], learning_rate);
        }

        double train_accuracy = evaluate(network, train);
        double test_accuracy = evaluate(network, test);

        printf("epoch %d: loss: %.4f train_accuracy: %.4f test_accuracy: %.4f\n", epoch, total_loss, train_accuracy, test_accuracy);
    }

    FreeNetwork(&network);
    FreeDataset(&train);
    FreeDataset(&test);

    return 0;
}