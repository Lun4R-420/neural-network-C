#ifndef WAV_H
#define WAV_H

typedef struct {
    // format chunk
    short int format;
    short int channels;
    int sample_rate;
    int byte_rate;
    short int block_align;
    short int bits_per_sample;

    // data chunk
    int num_samples;
    double *samples;
} Audio;

Audio LoadWav(const char *path);
void FreeAudio(Audio *a);

#endif