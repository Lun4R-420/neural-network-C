#ifndef AUDIO_DATASET_H
#define AUDIO_DATASET_H

#include "matrix.h"

typedef struct {
    int num;
    Matrix *features;
    Matrix *labels;
} AudioDataset;

AudioDataset LoadAudioDataset(const char *manifest_path, const char *audio_dir);
void FreeAudioDataset(AudioDataset *d);

#endif
