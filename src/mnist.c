#include <stdlib.h>
#include <stdio.h>
#include "mnist.h"

static unsigned int ReadBigEndian32(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return ((unsigned int) b[0] << 24) | ((unsigned int) b[1] << 16) | ((unsigned int) b[2] << 8) | ((unsigned int) b[3]);
}

Dataset LoadMnist(const char *images_path, const char *labels_path) {
    FILE *img_f = fopen(images_path, "rb");
    FILE *lbl_f = fopen(labels_path, "rb");

    ReadBigEndian32(img_f);
    unsigned int num = ReadBigEndian32(img_f);
    unsigned int rows = ReadBigEndian32(img_f);
    unsigned int cols = ReadBigEndian32(img_f);

    ReadBigEndian32(lbl_f);
    ReadBigEndian32(lbl_f);

    Dataset mnist;
    mnist.num = num;
    mnist.images = malloc(num * sizeof(Matrix));
    mnist.labels = malloc(num * sizeof(Matrix));

    unsigned char *img_buf = malloc(rows * cols);

    for (unsigned int i = 0; i < num; i++) {
        fread(img_buf, 1, rows * cols, img_f);
        Matrix image = CreateMatrix(rows * cols, 1);
        for (unsigned int p = 0; p < rows * cols; p++) {
            image.data[p] = img_buf[p] / 255.0;
        }
        mnist.images[i] = image;

        unsigned char label;
        fread(&label, 1, 1, lbl_f);
        Matrix label_onehot = CreateMatrix(10, 1);
        label_onehot.data[label] = 1.0;
        mnist.labels[i] = label_onehot;
    }

    free(img_buf);
    fclose(img_f);
    fclose(lbl_f);

    return mnist;
}

void FreeDataset(Dataset *d) {
    for (int i = 0; i < d->num; i++) {
        FreeMatrix(&d->images[i]);
        FreeMatrix(&d->labels[i]);
    }
    free(d->images);
    free(d->labels);
    
    d->images = NULL;
    d->labels = NULL;

    d->num = 0;
}