///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    complex_matmul.cpp
// Description: Perform matrix multiplication with complex values
//
// Note:        You are free to modify this code to implement your design.
///////////////////////////////////////////////////////////////////////////////

#include "complex.h"

void complex_matmul(
    complex_t MatA_DRAM[M_M][M_N], 
    complex_t MatB_DRAM[M_N][M_K], 
    complex_t MatC_DRAM[M_M][M_K]
)
{
#pragma HLS interface m_axi depth=1 port=MatA_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatB_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatC_DRAM offset=slave bundle=mem

#pragma HLS interface s_axilite port=return

    complex_t MatA[M_M][M_N];
    complex_t MatB[M_N][M_K];
    complex_t MatC[M_M][M_K];

    // Read in the data (Matrix A) from DRAM to BRAM
    MAT_A_ROWS:
    for(int i = 0; i < M_M; i++) {
        MAT_A_COLS:
        for(int j = 0; j < M_N; j++) {
            MatA[i][j] = MatA_DRAM[i][j];
        }
    }

    // Read in the data (Matrix B) from DRAM to BRAM
    MAT_B_ROWS:
    for(int i = 0; i < M_N; i++) {
        MAT_B_COLS:
        for(int j = 0; j < M_K; j++) {
            MatB[i][j] = MatB_DRAM[i][j];
        }
    }

    // Initialize product matrix C
    MAT_C_ROWS_INIT:
    for(int i = 0; i < M_M; i++) {
        MAT_C_COLS_INIT:
        for(int j = 0; j < M_K; j++) {
            MatC[i][j].real = 0;
            MatC[i][j].imag = 0;
        }
    }

    // Perform complex matrix multiplication 
    OUTER_ROWS:
    for(int i = 0; i < M_M; i++) {
        OUTER_COLS:
        for(int j = 0; j < M_K; j++) {
            INNER_ROW_COL:
            for(int p = 0; p < M_N; p++) {
                // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                int_t ac = MatA[i][p].real * MatB[p][j].real;
                int_t bd = MatA[i][p].imag * MatB[p][j].imag;
                int_t ad = MatA[i][p].real * MatB[p][j].imag;
                int_t bc = MatA[i][p].imag * MatB[p][j].real;
                
                MatC[i][j].real += ac - bd;
                MatC[i][j].imag += ad + bc;
            }
        }
    }

    // Write back the data from BRAM to DRAM
    MAT_C_ROWS:
    for(int i = 0; i < M_M; i++) {
        MAT_C_COLS:
        for(int j = 0; j < M_K; j++) {
            MatC_DRAM[i][j] = MatC[i][j];
        }
    }
}