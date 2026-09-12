#include <math.h>
#include <stdlib.h>
#include "audio_features.h"
#include "cplx.h"
#include "fft.h"

#define FRAME_SIZE 1024
#define HOP_SIZE 512

double FrameRms(double *samples, int start, int len) {
    double sum_sq = 0;
    for (int i = 0; i < len; i++) {
        sum_sq += samples[start + i] * samples[start + i];
    }
    return sqrt(sum_sq / len);
}

double FrameZcr(double *samples, int start, int len) {
    int signs_changes = 0;
    int prev_sign = samples[start] >= 0 ? 1 : -1;
    for (int i = 1; i < len; i++) {
        int sign = samples[start + i] >= 0 ? 1 : -1;
        if (sign != prev_sign) signs_changes++;
        prev_sign = sign;
    }
    return (double)signs_changes / (len - 1);
}

static double Mean(double *arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum / n;
}

static double StdDev(double *arr, int n, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = arr[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / n);
}

static double Max(double *arr, int n) {
    double max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

static double FrameSpectralCentroid(double *samples, int start, int len, int sample_rate) {
    Complex *x = malloc(len * sizeof(Complex));
    for (int i = 0; i < len; i++) {
        x[i].re = samples[start + i];
        x[i].im = 0.0;
    }

    Fft(x, len);

    double weighted_sum = 0.0;
    double magnitude_sum = 0.0;
    for (int k = 0; k <= len / 2; k++) {
        double mag = sqrt(x[k].re * x[k].re + x[k].im * x[k].im);
        double freq = (double)k * sample_rate / len;
        weighted_sum += freq * mag;
        magnitude_sum += mag;
    }

    free(x);

    if (magnitude_sum == 0.0) {
        return 0.0;
    }
    double centroid_hz = weighted_sum / magnitude_sum;
    double nyquist = sample_rate / 2.0;
    return centroid_hz / nyquist;
}

Matrix ExtractFeatures(Audio clip) {
    int num_frames = (clip.num_samples - FRAME_SIZE) / HOP_SIZE + 1;

    double *rms = malloc(num_frames * sizeof(double));
    double *zcr = malloc(num_frames * sizeof(double));
    double *centroid = malloc(num_frames * sizeof(double));

    for (int i = 0; i < num_frames; i++) {
        int start = i * HOP_SIZE;
        rms[i] = FrameRms(clip.samples, start, FRAME_SIZE);
        zcr[i] = FrameZcr(clip.samples, start, FRAME_SIZE);
        centroid[i] = FrameSpectralCentroid(clip.samples, start, FRAME_SIZE, clip.sample_rate);
    }

    double rms_mean = Mean(rms, num_frames);
    double rms_std = StdDev(rms, num_frames, rms_mean);
    double rms_max = Max(rms, num_frames);
    double zcr_mean = Mean(zcr, num_frames);
    double zcr_std = StdDev(zcr, num_frames, zcr_mean);
    double centroid_mean = Mean(centroid, num_frames);

    double delta_energy_sum = 0;
    for (int i = 1; i < num_frames; i++) {
        delta_energy_sum += fabs(rms[i] - rms[i - 1]);
    }
    double delta_energy_mean = delta_energy_sum / (num_frames - 1);

    free(rms);
    free(zcr);
    free(centroid);

    Matrix features = CreateMatrix(FEATURE_COUNT, 1);
    features.data[0] = rms_mean;
    features.data[1] = rms_std;
    features.data[2] = rms_max;
    features.data[3] = zcr_mean;
    features.data[4] = zcr_std;
    features.data[5] = delta_energy_mean;
    features.data[6] = centroid_mean;

    return features;
}