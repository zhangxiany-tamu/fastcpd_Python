// Stub implementations for Fortran printing functions (replacing ref_tseries_formats.c)
// These are used by the GARCH optimization Fortran code for debug output
// We provide minimal implementations that don't depend on R

#include <cstdio>

extern "C" {

// Fortran callable print routines (mostly stubs for GARCH optimizer output)
void cnlprt_C(char* msg, int* plen) {
    // Print message if needed (can be disabled for production)
    // printf("%.*s", *plen, msg);
    (void)msg;
    (void)plen;
}

void h30_(void) {
    // printf("\n");
}

void h40_(void) {
    // Header output - stub
}

void h70_(void) {
    // printf("\n INITIAL X\n");
}

void h80_(void) {
    // printf("\n     I     INITIAL X(I)        D(I)\n\n");
}

void h100s_C(int* i1, int* i2, double* d1, double* d2, double* d3, double* d4,
             double* d5, double* d6) {
    (void)i1; (void)i2; (void)d1; (void)d2; (void)d3; (void)d4; (void)d5; (void)d6;
}

void h100l_C(int* i1, int* i2, double* d1, double* d2, double* d3, double* d4,
             double* d5, double* d6, double* d7) {
    (void)i1; (void)i2; (void)d1; (void)d2; (void)d3; (void)d4; (void)d5; (void)d6; (void)d7;
}

void h110s_(int* i1, int* i2, double* d1, double* d2, double* d3, double* d4) {
    (void)i1; (void)i2; (void)d1; (void)d2; (void)d3; (void)d4;
}

void h110l_(int* i1, int* i2, double* d1, double* d2, double* d3, double* d4,
            double* d5) {
    (void)i1; (void)i2; (void)d1; (void)d2; (void)d3; (void)d4; (void)d5;
}

void h380_(int* i) {
    // printf(" ***** IV(1) =%5d *****\n", *i);
    (void)i;
}

void h400_(int* p, double* x, double* d) {
    (void)p; (void)x; (void)d;
}

void h410_(double* x) {
    (void)x;
}

void h420_(double* x) {
    (void)x;
}

void h450_(double* d1, double* d2, int* i1, int* i2, double* d3, double* d4) {
    (void)d1; (void)d2; (void)i1; (void)i2; (void)d3; (void)d4;
}

void h460_(int* i) {
    (void)i;
}

void h470_(int* i) {
    (void)i;
}

void h500_(int* p, double* x, double* d, double* g) {
    (void)p; (void)x; (void)d; (void)g;
}

}  // extern "C"
