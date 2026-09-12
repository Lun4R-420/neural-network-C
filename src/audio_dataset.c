#include <stdio.h>
#include <stdlib.h>
#include "audio_dataset.h"
#include "wav.h"
#include "audio_features.h"

static int CountLines(const char *path) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "LoadAudioDataset: cannot open %s\n", path);
        exit(1);
    }
    int count = 0;
    char line[512];
    while (fgets(line, sizeof(line), f) != NULL) {
        count++;
    }
    fclose(f);
    return count;
}

AudioDataset LoadAudioDataset(const char *manifest_path, const char *audio_dir) {
    int total = CountLines(manifest_path);

    AudioDataset d;
    d.num = total;
    d.features = malloc(total * sizeof(Matrix));
    d.labels = malloc(total * sizeof(Matrix));

    FILE *f = fopen(manifest_path, "r");
    char line[512];
    int i = 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        char filename[256];
        int label;
        if (sscanf(line, "%255[^,],%d", filename, &label) != 2) {
            continue;
        }

        char path[512];
        snprintf(path, sizeof(path), "%s/%s", audio_dir, filename);

        Audio clip = LoadWav(path);
        d.features[i] = ExtractFeatures(clip);
        FreeAudio(&clip);

        Matrix target = CreateMatrix(1, 1);
        target.data[0] = (double)label;
        d.labels[i] = target;

        i++;
    }

    fclose(f);
    return d;
}

void FreeAudioDataset(AudioDataset *d) {
    for (int i = 0; i < d->num; i++) {
        FreeMatrix(&d->features[i]);
        FreeMatrix(&d->labels[i]);
    }
    free(d->features);
    free(d->labels);
    d->features = NULL;
    d->labels = NULL;
    d->num = 0;
}
