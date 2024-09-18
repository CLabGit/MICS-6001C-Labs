///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    real.h
// Description: Header file for real matrix multiplication
//
// Note:        DO NOT MODIFY THIS CODE!
///////////////////////////////////////////////////////////////////////////////

#ifndef __REAL_H__
#define __REAL_H__

#include <stdio.h>
#include <stdlib.h>

#include <ap_int.h>

typedef ap_int<16> real_t;

#define M_N 150
#define M_M 100
#define M_K 200

void real_matmul( 
    real_t MatA_DRAM[M_M][M_N], 
    real_t MatB_DRAM[M_N][M_K], 
    real_t MatC_DRAM[M_M][M_K]
);

#endif
