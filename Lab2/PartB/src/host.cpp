#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include "conv.h"

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl_helper.hpp"

using namespace std;

// Convolution layer inputs, parameters, and reference output
float conv_layer_input_feature_map[3][736][1280];
float conv_layer_weights[64][3][7][7];
float conv_layer_bias[64];
float conv_layer_golden_output_feature_map[64][368][640];

fm_t  fixp_conv_layer_input_feature_map[3][736][1280];
wt_t  fixp_conv_layer_weights[64][3][7][7];
wt_t  fixp_conv_layer_bias[64];
fm_t  fixp_conv_layer_output_feature_map[64][368][640] = {0};

void read_bin_files() {
    // Input Feature Map
    ifstream ifs_conv_input("../bin/conv_input.bin", ios::in | ios::binary);
    ifs_conv_input.read((char*)(**conv_layer_input_feature_map), 3*736*1280*sizeof(float));
    ifs_conv_input.close();

    // Typecast to fixed-point 
    for(int c = 0; c < 3; c++)
        for(int i = 0; i < 736; i++)
            for(int j = 0; j < 1280; j++)
                fixp_conv_layer_input_feature_map[c][i][j] = (fm_t) conv_layer_input_feature_map[c][i][j];    
    
    // Weights
    ifstream ifs_conv_weights("../bin/conv_weights.bin", ios::in | ios::binary);
    ifs_conv_weights.read((char*)(***conv_layer_weights), 64*3*7*7*sizeof(float));
    ifs_conv_weights.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 64; f++)
        for(int c = 0; c < 3; c++)
            for(int m = 0; m < 7; m++)
                for(int n =0; n < 7; n++)
                    fixp_conv_layer_weights[f][c][m][n] = (wt_t) conv_layer_weights[f][c][m][n];
    
    // Bias
    ifstream ifs_conv_bias("../bin/conv_bias.bin", ios::in | ios::binary);
    ifs_conv_bias.read((char*)(conv_layer_bias), 64*sizeof(float));
    ifs_conv_bias.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 64; f++)
        fixp_conv_layer_bias[f] = (wt_t) conv_layer_bias[f];

    // Golden Output
    ifstream ifs_golden_output("../bin/conv_output.bin", ios::in | ios::binary);
    ifs_golden_output.read((char*)(**conv_layer_golden_output_feature_map), 64*368*640*sizeof(float));    
    ifs_golden_output.close();
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
    cl::Kernel krnl = xocl.get_kernel("tiled_conv");

    // Read reference inputs, parameters, and output
    read_bin_files();

    size_t input_size_bytes = sizeof(fm_t) * 3 * 736 * 1280;
    size_t weights_size_bytes = sizeof(wt_t) * 64 * 3 * 7 * 7;
    size_t bias_size_bytes = sizeof(wt_t) * 64;
    size_t output_size_bytes = sizeof(fm_t) * 64 * 368 * 640;

    std::cout << "Allocate Buffer in Global Memory\n";
    cl::Buffer input_buffer(xocl.get_context(), CL_MEM_READ_ONLY, input_size_bytes);
    cl::Buffer weights_buffer(xocl.get_context(), CL_MEM_READ_ONLY, weights_size_bytes);
    cl::Buffer bias_buffer(xocl.get_context(), CL_MEM_READ_ONLY, bias_size_bytes);
    cl::Buffer output_buffer(xocl.get_context(), CL_MEM_WRITE_ONLY, output_size_bytes);

    std::cout << "Copy data to device\n";
    q.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size_bytes, fixp_conv_layer_input_feature_map);
    q.enqueueWriteBuffer(weights_buffer, CL_TRUE, 0, weights_size_bytes, fixp_conv_layer_weights);
    q.enqueueWriteBuffer(bias_buffer, CL_TRUE, 0, bias_size_bytes, fixp_conv_layer_bias);

    std::cout << "Set kernel arguments\n";
    krnl.setArg(0, input_buffer);
    krnl.setArg(1, weights_buffer);
    krnl.setArg(2, bias_buffer);
    krnl.setArg(3, output_buffer);

    std::cout << "Execution of the kernel\n";
    cl::Event event;
    q.enqueueTask(krnl, NULL, &event);
    event.wait();

    std::cout << "Read back computation results\n";
    q.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size_bytes, fixp_conv_layer_output_feature_map);

    // Compute Mean-Squared-Error
    long double mse = 0.0;
    for(int f = 0; f < 64; f++) {
        for(int i = 0; i < 368; i++) {
            std::cout << conv_layer_golden_output_feature_map[f][i][0] << " " << fixp_conv_layer_output_feature_map[f][i][0] << " " << f << " " << i << std::endl;
            for(int j = 0; j < 640; j++) {
                mse += std::pow((conv_layer_golden_output_feature_map[f][i][j] 
                                 -(float) fixp_conv_layer_output_feature_map[f][i][j]), 2);
            }
        }
    }
    std::cout << conv_layer_golden_output_feature_map[0][125][480] << " " << fixp_conv_layer_output_feature_map[0][125][480] << std::endl;
    std::cout << conv_layer_golden_output_feature_map[63][367][0] << " " << fixp_conv_layer_output_feature_map[63][367][0] << std::endl;
    mse = mse / (64 * 368 * 640);

    std::cout << "\nOutput MSE:  " << mse << std::endl;
    
    std::cout << "----------------------------------------" << std::endl;
    if(mse > 0 && mse < std::pow(10,-2)) {
        std::cout << "Fixed-point Simulation SUCCESSFUL!!!" << std::endl;
    } else {
        std::cout << "Fixed-point Simulation FAILED :(" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}