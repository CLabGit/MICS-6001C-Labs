/* Testbench for Csim */

#include "real.h"

int main()
{
    // Declare matrices
    real_t MatA_tb[M_M][M_N];
    real_t MatB_tb[M_N][M_K];
    real_t MatC_tb[M_M][M_K];
    real_t MatC_expected[M_M][M_K];

    // Generate Matrix A with random values
    for(int i = 0; i < M_M; i++) {
        for(int j = 0; j < M_N; j++) {
            MatA_tb[i][j] = rand() % 50;
        }
    }

    // Generate Matrix B with random values
    for(int i = 0; i < M_N; i++) {
        for(int j = 0; j < M_K; j++) {
            MatB_tb[i][j] = rand() % 50;
        }
    }

    // Initialize Matrix C 
    for(int i = 0; i < M_M; i++) {
        for(int j = 0; j < M_K; j++) {
            MatC_tb[i][j] = 0;
        }
    }

    // Call DUT
    real_matmul(MatA_tb, MatB_tb, MatC_tb);

    // Expected value for Matrix C
    // To make sure your optimizations do not change the functionality
    for(int i = 0; i < M_M; i++) {
        for(int j = 0; j < M_K; j++) {
            
            MatC_expected[i][j] = 0;
            for(int p = 0; p < M_N; p++) {
                MatC_expected[i][j] += MatA_tb[i][p] * MatB_tb[p][j];
            }

        }
    }

    // Verify functional correctness before synthesizing
    int passed = 1;
    for(int i = 0; i < M_M; i++) {
        for(int j = 0; j < M_K; j++) {
            if(MatC_tb[i][j] != MatC_expected[i][j]) {
                printf("Mismatch at MatC[%d][%d]: Expected: %hi \t Actual: %hi\n", 
                        i, j, MatC_expected[i][j], MatC_tb[i][j]);
                passed = 0;
            }
        }
    }
    
    if(passed) {
        printf("-----------------------------------\n");
        printf("|         TEST PASSED!            |\n");
        printf("-----------------------------------\n");
    }
    else {
        printf("-----------------------------------\n");
        printf("|         TEST FAILED :(          |\n");
        printf("-----------------------------------\n");
    }

    return !passed;
}
