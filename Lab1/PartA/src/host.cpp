#include <iostream>
#include <cstring>
#include <chrono>
#include "real.h"

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl_helper.hpp"
#define DATA_SIZE (M_M * M_N * M_K)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xclbin_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    xilinx::example_utils::XilinxOclHelper xocl;
    xocl.initialize(binaryFile);
    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl = xocl.get_kernel("real_matmul");

    size_t matrix_a_size_bytes = sizeof(real_t) * M_M * M_N;
    size_t matrix_b_size_bytes = sizeof(real_t) * M_N * M_K;
    size_t matrix_c_size_bytes = sizeof(real_t) * M_M * M_K;

    std::cout << "Allocate Buffer in Global Memory\n";
    cl::Buffer a_to_device(xocl.get_context(), CL_MEM_READ_ONLY, matrix_a_size_bytes);
    cl::Buffer b_to_device(xocl.get_context(), CL_MEM_READ_ONLY, matrix_b_size_bytes);
    cl::Buffer c_from_device(xocl.get_context(), CL_MEM_WRITE_ONLY, matrix_c_size_bytes);

    real_t* a = new real_t[M_M * M_N];
    real_t* b = new real_t[M_N * M_K];
    real_t* c = new real_t[M_M * M_K];
    real_t MatC_expected[M_M][M_K];

    std::cout << "Populating buffer inputs\n";
    for (int i = 0; i < M_M; i++) {
        for (int j = 0; j < M_N; j++) {
            a[i * M_N + j] = rand() % 50;
        }
    }

    for (int i = 0; i < M_N; i++) {
        for (int j = 0; j < M_K; j++) {
            b[i * M_K + j] = rand() % 50;
        }
    }

    for (int i = 0; i < M_M; i++) {
        for (int j = 0; j < M_K; j++) {
            MatC_expected[i][j] = 0;
            for (int p = 0; p < M_N; p++) {
                MatC_expected[i][j] += a[i * M_N + p] * b[p * M_K + j];
            }
        }
    }

    std::cout << "Copy data to device\n";
    q.enqueueWriteBuffer(a_to_device, CL_TRUE, 0, matrix_a_size_bytes, a);
    q.enqueueWriteBuffer(b_to_device, CL_TRUE, 0, matrix_b_size_bytes, b);

    std::cout << "Set kernel arguments\n";
    krnl.setArg(0, a_to_device);
    krnl.setArg(1, b_to_device);
    krnl.setArg(2, c_from_device);
    // krnl.setArg(3, M_M);
    // krnl.setArg(4, M_N);
    // krnl.setArg(5, M_K);

    std::cout << "Execution of the kernel\n";
    cl::Event event;
    q.enqueueTask(krnl, NULL, &event);
    event.wait();

    std::cout << "Read back computation results\n";
    q.enqueueReadBuffer(c_from_device, CL_TRUE, 0, matrix_c_size_bytes, c);

    bool match = true;
    for (int i = 0; i < M_M; i++) {
        for (int j = 0; j < M_K; j++) {
            if (c[i * M_K + j] != MatC_expected[i][j]) {
                match = false;
                std::cerr << "Mismatch at (" << i << ", " << j << "): " << c[i * M_K + j] << " != " << MatC_expected[i][j] << std::endl;
                break;
            }
        }
    }

    if (match) {
        std::cout << "-----------------------------------\n";
        std::cout << "|         TEST PASSED!            |\n";
        std::cout << "-----------------------------------\n";
    } else {
        std::cout << "-----------------------------------\n";
        std::cout << "|         TEST FAILED :(          |\n";
        std::cout << "-----------------------------------\n";
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
