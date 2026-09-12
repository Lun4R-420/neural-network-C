#include <stdlib.h>
#include <math.h>
#include "fft.h"

void Fft(Complex *x, int n) {
    if (n <= 1) {
        return;
    }

    Complex *even = malloc((n / 2) * sizeof(Complex));
    Complex *odd = malloc((n / 2) * sizeof(Complex));

    for (int i = 0; i < n / 2; i++) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    Fft(even, n / 2);
    Fft(odd, n / 2);

    for (int k = 0; k < n / 2; k++) {
        double angle = -2.0 * M_PI * k / n;
        Complex twiddle = { cos(angle), sin(angle) };
        Complex t = Multiply(twiddle, odd[k]);

        x[k] = AddComplex(even[k], t);
        x[k + n / 2] = SubComplex(even[k], t);
    }

    free(even);
    free(odd);
}
