#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wav.h"

Audio LoadWav(const char *path) {
    FILE *audio_file = fopen(path, "rb");
    Audio audio;

    char buffer[12];
    fread(buffer, 1, 12, audio_file);

    char chunk_id[4];
    fread(chunk_id, 1, 4, audio_file);
    if (memcmp(chunk_id, "fmt ", 4) != 0) {
        fprintf(stderr, "expected fmt chunk\n");
        exit(1);
    }

    int chunk_size;
    fread(&chunk_size, 4, 1, audio_file);

    short int audio_format;
    fread(&audio_format, 1, 2, audio_file);
    audio.format = audio_format;

    short int channels;
    fread(&channels, 1, 2, audio_file);
    audio.channels = channels;

    int sample_rate;
    fread(&sample_rate, 1, 4, audio_file);
    audio.sample_rate = sample_rate;

    int byte_rate;
    fread(&byte_rate, 1, 4, audio_file);
    audio.byte_rate = byte_rate;

    short int block_align;
    fread(&block_align, 1, 2, audio_file);
    audio.block_align = block_align;

    short int bits_per_sample;
    fread(&bits_per_sample, 1, 2, audio_file);
    audio.bits_per_sample = bits_per_sample;

    fread(chunk_id, 1, 4, audio_file);
    if (memcmp(chunk_id, "data", 4) != 0) {
        fprintf(stderr, "expected data chunk\n");
        exit(1);
    }

    fread(&chunk_size, 1, 4, audio_file);
    int num_samples = chunk_size / 2; // each sample is 2 bytes
    audio.num_samples = num_samples;
    audio.samples = malloc(num_samples * sizeof(double));
    for (int i = 0; i < num_samples; i++) {
        short raw;
        fread(&raw, 2, 1, audio_file);
        audio.samples[i] = raw / 32768.0;
    }

    fclose(audio_file);
    return audio;
}

void FreeAudio(Audio *a) {
    a->format = 0;
    a->sample_rate = 0;
    a->byte_rate = 0;
    a->bits_per_sample = 0;
    
    free(a->samples);
    a->samples = NULL;
    a->num_samples = 0;
}