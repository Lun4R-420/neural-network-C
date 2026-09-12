#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "audio_dataset.h"
#include "audio_features.h"

static void ShuffleIndices(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

static double Evaluate(Network network, AudioDataset d, int *indices, int count) {
    int correct = 0;
    for (int k = 0; k < count; k++) {
        int i = indices[k];
        Matrix output = NetworkForward(network, d.features[i]);
        int predicted = output.data[0] > 0.5;
        int actual = d.labels[i].data[0] > 0.5;
        if (predicted == actual) {
            correct++;
        }
        FreeMatrix(&output);
    }
    return (double)correct / count;
}

int main() {
    srand(42);

    AudioDataset d = LoadAudioDataset("data/audio_manifest.csv", "data/audio");
    printf("loaded %d clips\n", d.num);

    int *indices = malloc(d.num * sizeof(int));
    for (int i = 0; i < d.num; i++) {
        indices[i] = i;
    }
    ShuffleIndices(indices, d.num);

    int train_count = (int)(d.num * 0.8);
    int test_count = d.num - train_count;
    int *train_indices = indices;
    int *test_indices = indices + train_count;

    int test_anomaly = 0;
    for (int k = 0; k < test_count; k++) {
        if (d.labels[test_indices[k]].data[0] > 0.5) test_anomaly++;
    }
    printf("test set: %d anomaly, %d normal (majority baseline: %.4f)\n",
           test_anomaly, test_count - test_anomaly,
           (double)(test_count - test_anomaly > test_anomaly ? test_count - test_anomaly : test_anomaly) / test_count);

    int *train_anomaly_idx = malloc(train_count * sizeof(int));
    int *train_normal_idx = malloc(train_count * sizeof(int));
    int n_anomaly = 0, n_normal = 0;
    for (int k = 0; k < train_count; k++) {
        int i = train_indices[k];
        if (d.labels[i].data[0] > 0.5) {
            train_anomaly_idx[n_anomaly++] = i;
        } else {
            train_normal_idx[n_normal++] = i;
        }
    }
    printf("train set (before balancing): %d anomaly, %d normal\n", n_anomaly, n_normal);

    int repeat_factor = n_normal / n_anomaly;
    int balanced_count = n_normal + n_anomaly * repeat_factor;
    int *balanced_indices = malloc(balanced_count * sizeof(int));
    int pos = 0;
    for (int k = 0; k < n_normal; k++) {
        balanced_indices[pos++] = train_normal_idx[k];
    }
    for (int r = 0; r < repeat_factor; r++) {
        for (int k = 0; k < n_anomaly; k++) {
            balanced_indices[pos++] = train_anomaly_idx[k];
        }
    }
    ShuffleIndices(balanced_indices, balanced_count);
    printf("train set (after balancing): %d anomaly, %d normal (total %d)\n",
           n_anomaly * repeat_factor, n_normal, balanced_count);

    free(train_anomaly_idx);
    free(train_normal_idx);

    int layer_sizes[] = {FEATURE_COUNT, 4, 1};
    Network network = CreateNetwork(3, layer_sizes);

    int epochs = 300;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int k = 0; k < balanced_count; k++) {
            int i = balanced_indices[k];
            total_loss += NetworkTrainStep(&network, d.features[i], d.labels[i], learning_rate);
        }

        if (epoch % 20 == 0 || epoch == epochs - 1) {
            double train_acc = Evaluate(network, d, train_indices, train_count);
            double test_acc = Evaluate(network, d, test_indices, test_count);
            printf("epoch %d loss %.4f train_acc %.4f test_acc %.4f\n",
                   epoch, total_loss / balanced_count, train_acc, test_acc);
        }
    }

    free(balanced_indices);
    free(indices);
    FreeNetwork(&network);
    FreeAudioDataset(&d);

    return 0;
}
