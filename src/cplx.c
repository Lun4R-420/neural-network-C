#include "cplx.h"

Complex AddComplex(Complex a, Complex b) {
    Complex result;
    result.re = a.re + b.re;
    result.im = a.im + b.im;
    
    return result;
}

Complex SubComplex(Complex a, Complex b) {
    Complex result;
    result.re = a.re - b.re;
    result.im = a.im - b.im;

    return result;
}

Complex Multiply(Complex a, Complex b) {
    Complex result;
    result.re = a.re * b.re - a.im * b.im;
    result.im = a.re * b.im + a.im * b.re;

    return result;
}