#ifndef CPLX_H
#define CPLX_H

typedef struct {
    double re;
    double im;
} Complex;

Complex AddComplex(Complex a, Complex b);
Complex SubComplex(Complex a, Complex b);
Complex Multiply(Complex a, Complex b);

#endif
