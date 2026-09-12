#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include "matrix.h"
#include "wav.h"

#define FEATURE_COUNT 7

Matrix ExtractFeatures(Audio clip);

#endif
