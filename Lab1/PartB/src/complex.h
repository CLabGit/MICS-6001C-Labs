///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    complex.h
// Description: Header file for complex matrix multiplication
//
// Note:        DO NOT MODIFY THIS CODE!
///////////////////////////////////////////////////////////////////////////////
#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include <ap_int.h>

typedef ap_int<16> int_t;

typedef struct complex_t {
    int_t real;
    int_t imag;
} complex_t;

#define M_M 100
#define M_N 150
#define M_K 200

void complex_matmul ( 
    complex_t MatA_DRAM[M_M][M_N], 
    complex_t MatB_DRAM[M_N][M_K], 
    complex_t MatC_DRAM[M_M][M_K]
);

using namespace std;

#endif
