/* Testbench for Complex Matrix Multiplication */

#include <iostream>
#include <fstream>
#include "complex.h"

void loadMatrix(const std::string& filename, int rows, int cols, complex_t* Mat) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int real, imag;
            file >> real >> imag;
            Mat[i * cols + j].real = real;
            Mat[i * cols + j].imag = imag;
        }
    }
}

int main() {

    complex_t MatA_tb[M_M * M_N];
    complex_t MatB_tb[M_N * M_K];
    complex_t MatC_tb[M_M * M_K];
    complex_t MatC_expected[M_M * M_K];

    loadMatrix("MatA.txt", M_M, M_N, MatA_tb);
    loadMatrix("MatB.txt", M_N, M_K, MatB_tb);
    loadMatrix("MatC.txt", M_M, M_K, MatC_expected);

    complex_t MatA_DRAM[M_M][M_N];
    complex_t MatB_DRAM[M_N][M_K];
    complex_t MatC_DRAM[M_M][M_K];

    for (int i = 0; i < M_M; i++) {
        for (int j = 0; j < M_N; j++) {
            MatA_DRAM[i][j] = MatA_tb[i * M_N + j];
        }
    }

    for (int i = 0; i < M_N; i++) {
        for (int j = 0; j < M_K; j++) {
            MatB_DRAM[i][j] = MatB_tb[i * M_K + j];
        }
    }

    for (int i = 0; i < M_M; i++) {
        for (int j = 0; j < M_K; j++) {
            MatC_DRAM[i][j].real = 0;
            MatC_DRAM[i][j].imag = 0;
        }
    }

    complex_matmul(MatA_DRAM, MatB_DRAM, MatC_DRAM);

    bool passed = 1;
    for (int i = 0; i < M_M; i++) {
        for (int j = 0; j < M_K; j++) {
            if (MatC_DRAM[i][j].real != MatC_expected[i * M_K + j].real || 
                MatC_DRAM[i][j].imag != MatC_expected[i * M_K + j].imag) {
                std::cout << "Mismatch at MatC[" << i << "][" << j << "]: Expected: (" 
                          << MatC_expected[i * M_K + j].real << " + " << MatC_expected[i * M_K + j].imag << "j) \t Actual: (" 
                          << MatC_DRAM[i][j].real << " + " << MatC_DRAM[i][j].imag << "j)" << std::endl;
                passed = 0;
            }
        }
    }

    if (passed) {
        std::cout << "-----------------------------------\n";
        std::cout << "|         TEST PASSED!            |\n";
        std::cout << "-----------------------------------\n";
    } else {
        std::cout << "-----------------------------------\n";
        std::cout << "|         TEST FAILED :(          |\n";
        std::cout << "-----------------------------------\n";
    }

    return !passed;
}