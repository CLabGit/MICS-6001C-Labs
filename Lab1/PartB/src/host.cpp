#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include "complex.h"

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl_helper.hpp"

#define DATA_SIZE (M_M * M_N * M_K)

void loadMatrix(std::string filename, int numRows, int numCols, complex_t* Mat_tb) {
    int i = 0, j = 0, t = 0;
    int_t real, imag;

    std::ifstream Mat(filename);

    while(Mat >> real >> imag) {
        i = t / numCols;
        j = t % numCols;
        Mat_tb[i * numCols + j].real = real;
        Mat_tb[i * numCols + j].imag = imag;
        t++;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xclbin_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    xilinx::example_utils::XilinxOclHelper xocl;
    xocl.initialize(binaryFile);
    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl = xocl.get_kernel("complex_matmul");

    size_t matrix_a_size_bytes = sizeof(complex_t) * M_M * M_N;
    size_t matrix_b_size_bytes = sizeof(complex_t) * M_N * M_K;
    size_t matrix_c_size_bytes = sizeof(complex_t) * M_M * M_K;

    std::cout << "Allocate Buffer in Global Memory\n";
    cl::Buffer a_to_device(xocl.get_context(), CL_MEM_READ_ONLY, matrix_a_size_bytes);
    cl::Buffer b_to_device(xocl.get_context(), CL_MEM_READ_ONLY, matrix_b_size_bytes);
    cl::Buffer c_from_device(xocl.get_context(), CL_MEM_WRITE_ONLY, matrix_c_size_bytes);

    complex_t* MatA_tb = new complex_t[M_M * M_N];
    complex_t* MatB_tb = new complex_t[M_N * M_K];
    complex_t* MatC_tb = new complex_t[M_M * M_K];
    complex_t* MatC_expected = new complex_t[M_M * M_K];

    std::cout << "Loading input matrices and expected output matrix\n";
    loadMatrix("MatA.txt", M_M, M_N, MatA_tb);
    loadMatrix("MatB.txt", M_N, M_K, MatB_tb);
    loadMatrix("MatC.txt", M_M, M_K, MatC_expected);

    std::cout << "Copy data to device\n";
    q.enqueueWriteBuffer(a_to_device, CL_TRUE, 0, matrix_a_size_bytes, MatA_tb);
    q.enqueueWriteBuffer(b_to_device, CL_TRUE, 0, matrix_b_size_bytes, MatB_tb);

    std::cout << "Set kernel arguments\n";
    krnl.setArg(0, a_to_device);
    krnl.setArg(1, b_to_device);
    krnl.setArg(2, c_from_device);

    std::cout << "Execution of the kernel\n";
    cl::Event event;
    q.enqueueTask(krnl, NULL, &event);
    event.wait();

    std::cout << "Read back computation results\n";
    q.enqueueReadBuffer(c_from_device, CL_TRUE, 0, matrix_c_size_bytes, MatC_tb);

    // Verify functional correctness
    bool passed = true;
    for(int i = 0; i < M_M; i++) {
        for(int j = 0; j < M_K; j++) {
            if((MatC_tb[i * M_K + j].real != MatC_expected[i * M_K + j].real) || 
               (MatC_tb[i * M_K + j].imag != MatC_expected[i * M_K + j].imag)) {
                std::cout << "Mismatch at MatC[" << i << "][" << j << "]: Expected: (" 
                          << MatC_expected[i * M_K + j].real << " + " << MatC_expected[i * M_K + j].imag << "j) \t Actual: (" 
                          << MatC_tb[i * M_K + j].real << " + " << MatC_tb[i * M_K + j].imag << "j)" << std::endl;
                passed = false;
            }
        }
    }
    
    if(passed) {
        std::cout << "-----------------------------------\n";
        std::cout << "|         TEST PASSED!            |\n";
        std::cout << "-----------------------------------\n";
    } else {
        std::cout << "-----------------------------------\n";
        std::cout << "|         TEST FAILED :(          |\n";
        std::cout << "-----------------------------------\n";
    }

    delete[] MatA_tb;
    delete[] MatB_tb;
    delete[] MatC_tb;
    delete[] MatC_expected;

    return 0;
}