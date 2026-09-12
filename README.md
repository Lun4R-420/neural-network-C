# Neural Network Engine (C, from scratch)

I built a neural network from zero in C, no ML frameworks, no linear algebra library, just `stdio`/`stdlib`/`math`. The reason for C isn't nostalgia, it's that the end goal is running this same inference code on a constrained embedded target (ARM Cortex-M) later, not just a PC.

The actual task I'm using it for is audio anomaly detection for home security: given a short clip, decide if it's an "anomaly" (glass breaking, a siren) or normal background noise (vacuum cleaner, dog barking, footsteps, a door knock, a clock ticking).

## What's in here

- `matrix.c`/`matrix.h`: a `Matrix` struct and the basic ops on it (multiply, add, transpose, elementwise), all on flat arrays.
- `Network`: a generalized feedforward network, any number of layers, any sizes. Forward pass plus full backprop in `NetworkTrainStep`.
- `mnist.c`: loads the MNIST idx-ubyte files (they're big-endian, which is a fun contrast to WAV). I used MNIST purely to check the engine actually works before pointing it at the real task.
- `wav.c`: a WAV/RIFF parser, written by hand, no libsndfile. Reads the `fmt` and `data` chunks, decodes 16-bit PCM.
- `cplx.c`/`fft.c`: complex numbers and a recursive Cooley-Tukey FFT, also from scratch, used for the spectral feature.
- `audio_features.c`: turns a raw clip into 7 numbers, RMS (mean/std/max), zero-crossing rate (mean/std), how fast the energy jumps between frames, and a spectral centroid from the FFT.
- `audio_dataset.c`: reads the ESC-50 manifest and turns the listed clips into feature/label pairs.

## Why these features specifically

First pass was just RMS and zero-crossing rate, and it wasn't enough, the network plateaued barely above just guessing the majority class every time. Adding the spectral centroid (roughly: where in the frequency range the sound's energy sits) is what actually moved the needle, since a glass-breaking transient and a steady vacuum-cleaner hum look pretty different in frequency even when their raw loudness is similar.

## Results

MNIST was just a sanity check, not the point, but for the record: 96%+ test accuracy on a 784-128-10 network.

For the real task, ESC-50 subset (280 clips, imbalanced toward normal sounds so the majority-class baseline is ~68%): after rebalancing the classes and adding the spectral feature, we're at 80% test accuracy.

## Building

No dependencies besides libc and libm:

```bash
git clone https://github.com/Lun4R-420/neural-network-C.git
cd neural-network-C
```

```bash
gcc -O2 src/matrix.c src/wav.c src/cplx.c src/fft.c src/audio_features.c src/audio_dataset.c src/train_audio.c -o build/train_audio -lm
./build/train_audio
```

Datasets aren't committed (see `.gitignore`). MNIST files come from [this mirror](https://github.com/cvdfoundation/mnist), audio clips from [ESC-50](https://github.com/karolpiczak/ESC-50).

## What's left

Right now it only runs on a PC. Next step is exporting the trained weights and porting just the inference path (no training, no backprop) to an actual ARM Cortex-M board.
