///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Filename:    real_matmul.cpp
// Description: Perform matrix multiplication with real values
//
// Note:        You are free to modify this code to optimize your design.
///////////////////////////////////////////////////////////////////////////////

#include "real.h"

void real_matmul( 
    real_t MatA_DRAM[M_M][M_N], 
    real_t MatB_DRAM[M_N][M_K], 
    real_t MatC_DRAM[M_M][M_K])
{
#pragma HLS interface m_axi depth=1 port=MatA_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatB_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatC_DRAM offset=slave bundle=mem

#pragma HLS interface s_axilite port=return
    
    real_t MatA[M_M][M_N];
    real_t MatB[M_N][M_K];
    real_t MatC[M_M][M_K];

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
            MatC[i][j] = 0;
        }
    }

    // Perform matrix multiplication 
    OUTER_ROWS:
    for(int i = 0; i < M_M; i++) {
        OUTER_COLS:
        for(int j = 0; j < M_K; j++) {
            INNER_ROW_COL:
            for(int p = 0; p < M_N; p++) {
                MatC[i][j] += MatA[i][p] * MatB[p][j];
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
