#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main() {
    srand(42);

    int layer_sizes[] = {2, 2, 1};
    Network network = CreateNetwork(3, layer_sizes);

    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4] = {0, 1, 1, 0};

    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++) {
            Matrix X = CreateMatrix(2, 1);
            X.data[0] = inputs[i][0];
            X.data[1] = inputs[i][1];
            Matrix y = CreateMatrix(1, 1);
            y.data[0] = targets[i];

            total_loss += NetworkTrainStep(&network, X, y, 0.5);

            FreeMatrix(&X);
            FreeMatrix(&y);
        }

        if (epoch % 1000 == 0) {
            printf("epoch %d loss %f\n", epoch, total_loss);
        }
    }

    for (int i = 0; i < 4; i++) {
        Matrix X = CreateMatrix(2, 1);
        X.data[0] = inputs[i][0];
        X.data[1] = inputs[i][1];

        Matrix output = NetworkForward(network, X);
        printf("(%.0f,%.0f) -> %f\n", inputs[i][0], inputs[i][1], output.data[0]);

        FreeMatrix(&X);
        FreeMatrix(&output);
    }

    FreeNetwork(&network);

    return 0;
}
