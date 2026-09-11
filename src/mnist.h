#ifndef MNIST_H
#define MNIST_H

#include "matrix.h"

typedef struct {
    int num;
    Matrix *images;
    Matrix *labels;
} Dataset;

Dataset LoadMnist(const char *images_path, const char *labels_path);
void FreeDataset(Dataset *d);

#endif
