# Lab 1 Instructions

The goal of this lab is to design accelerators for matrix multiplication. There are two parts to this lab:
- In Part A, you will optimize an existing implementation of matrix multiplication with real values. 
- In Part B, you will implement matrix multiplication with complex values and optimize it. 

In either part, we will multiply a $100\*150$ matrix with a $150\*200$ matrix to get a $100\*200$ matrix product and accelerate the design.


## Part A: Real Matrix Multiplication (30 points)
Matrix multiplication is at the heart of virtually all deep learning applications and has high scope for parallelism. Compared to a trivial implementation, the amount of parallelism that you can exploit is constrained by the hardware resources and how well your code is optimized. 

In this part of the lab, you are provided with the trivial (simplest) implementation along with a testbench to verify functionality. 
You need to optimize the design to achieve at least **10x** speed-up while ensuring the conrrect deployment on FPGA boards.
- `PartA/real_matmul.cpp` contains the matrix multiplier, which is the top module of the accelerator. This is what you want to modify.
- `PartA/host.cpp` is the testbench. Do not change this file.

**Reference Material**: Ryan Kastner et al., [Parallel Programming for FPGAs](https://github.com/KastnerRG/pp4fpgas/raw/gh-pages/main.pdf), Chapter 7.

## Part B: Complex Matrix Multiplication (70 points)
Many applications in scientific computing and signal processing require working with not just the magnitude but also the phase. This information is aptly captured using complex numbers. Let's build on Part A and develop an accelerator to perform matrix multiplication with complex numbers.

In this part of the lab, you are **not** provided with the trivial implementation. You need to implement the design first (can be trivial), ensure its functional correctness with the testbench provided, and then optimize the design to achieve at least **10x** speed-up while ensuring the resource utilization are all under 100%.
- `PartB/complex_matmul.cpp` is the top module of the accelerator that you need to modify.
- `PartB/host.cpp` is the testbench. Do not change this file.

**Reference Material**: [Complex Matrix Multiplication](https://mathworld.wolfram.com/ComplexMatrix.html)

## How to Run the Codes

In our HLS tutorial, for faster development, we used `Makefile` script for simulation, synthesis and program.

### Step 1: C Simulation

In `PartA` or `Part B`, go to the relevant folder and type `make csim`. After a compilation, the generated executable file is located in the `./build_hls/hls/csim/csim.exe`, and you can also run it directly to view the results.

*Important*: Please run C simulation after every change you make to your top function. Cannot stress this enough!

### Step 2: C Synthesis

To run C synthesis, you can use `Makefile` again. Go to the relevant folder (`PartA` or `PartB`) and type `make csynth`.

Once synthesis completes (this shouldn't take more than a minute or two!), you can either open the GUI to read the reports, or you can find the reports under `./buld_hls/hls/syn/report/`. The reports with `.rpt` extensions can be opened using text editors.

We recommend using `csynth.rpt` to assess the overall latency and resource utilization (BRAM, DSP, FF and LUT). You can check other reports for analysis purposes.

### Step 3: Hardware Emulation

The kernel requires a host to declare and run, so go to the relevant folder (`PartA` or `PartB`) and type `make build TARGET=hw_emu` to create a kernel targeting `hw_emu` and compile an executable file for the host based on OpenCL API.

Waiting for a period of time (around 30 mins), the xclbin file for the `hw` will be completed. Next, use `make run TARGET=hw_emu` to run the kernel on the FPGA board. This internally runs:

```
XCL_EMULATION_MODE=hw_emu ./host_name.exe kernel_name.xclbin
```
Harware Emulation has been an important way to debug your design, since it generates RTL from the accelerator sources and run RTL simulation along with the application layer code. Users can view simulation waveforms by enabling the following switch in the xrt.ini file. But, ensure you have X-server opened for displaying the waveforms. 

```
[Emulation]
debug_mode=gui
```

### Step 3: Test on board

The kernel requires a host to declare and run, so go to the relevant folder (`PartA` or `PartB`) and type `make build` to create a kernel targeting `hw` and compile an executable file for the host based on OpenCL API.

Waiting for a period of time (possibly 1-3 hours), the xclbin file for the `hw` will be completed. Next, use `make run` to run the kernel on the FPGA board. This internally runs:

```
./host_name.exe kernel_name.xclbin
```

## What to Submit for this Lab

### Part A
1. (20 points) Optimized source code and top-level synthesis report:
    - `real_matmul.cpp`
    - `real_csynth.rpt` (Please rename `csynth.rpt` to `real_csynth.rpt` before submitting)

2. (10 points) A brief report including:
    - The baseline latency and resource utilization
    - The optimized latency you achieved; how much speed up you gained?
    - The resource utilization
    - What are the main techniques you adopted?

### Part B
1. (50 points) Optimized source code and top-level synthesis report:
    - `complex_matmul.cpp`
    - `complex_csynth.rpt` (Please rename `csynth.rpt` to `complex_csynth.rpt` before submitting)

2. (20 points) A brief report including:
    - The baseline latency and resource utilization
    - The optimized latency; how much speed up you gained compared to baseline?
    - The resource utilization (after optimization)
    - What are the main techniques you adopted?

*Note*: Please combine your Part A and Part B reports in a single file and submit `Lab1_Report_<Name>.pdf`. There is no template to follow, however, you are expected to write your report like a research paper.


## Academic Integrity and Honor Code
You are required to follow HKUST(GZ)'s [Honor Code](https://fytgs.hkust-gz.edu.cn/handbooks/handbook-for-research-postgraduate-studies) and submit your original work in this lab, especially for Part A. You can discuss this assignment with other classmates but you should code your assignment individually. You are **NOT** allowed to see the code of (or show your code to) other students.

Furthermore, you should **NOT** be looking at the solutions provided in the previous iteration of this course. We will be using the Stanford MOSS tool to detect plagiarism. When there is reasonably clear evidence of a violation, a referral to the Office of the Dean of Students will occur, and all hearings and other resulting procedures will be followed to completion.

## Submission Guideline

Submission: on Canvas for course students, via email (to TA) for Special Problems students

Due date: **TBD**.


## Acknowledgements

This lab is adapted from ECE 8893 - Parallel Programming on FPGAs at Georgia Tech.

## Grading Rubric
$$
Speedup = \frac{Baseline\ Latency}{Optimized\ Latency}
$$

### Part A.1 (20 points)

> simulationTestPass &rarr; +5 points   
> **if** (simulationTestPass):  
> &nbsp;&nbsp;&nbsp;&nbsp; **if**(speedup &geq; 10x), +7 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else if**(2x &leq; speedup < 10x), +4 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else**, +2 points  
>   
> &nbsp;&nbsp;&nbsp;&nbsp; **for** resource **in** [BRAM, DSP, FF, LUT]:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+2 points **if** utilization &leq; 100%  

### Part A.2 (10 points)

> Missing or incomplete information &rarr; -1 point for each question  
> Report values different from the ones achieved in **Part A.1**  &rarr; -2 points each   
> Insufficient description of technique(s) adopted &rarr; -2 points   

### Part B.1 (50 points)

> simulationTestPass &rarr; +15 points   
> **if** (simulationTestPass):  
> &nbsp;&nbsp;&nbsp;&nbsp; **if**(speedup &geq; 10x), +15 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else if**(2x &leq; speedup < 10x), +10 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else**, +5 points  
>   
> &nbsp;&nbsp;&nbsp;&nbsp; **for** resource **in** [BRAM, DSP, FF, LUT]:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+5 points **if** utilization &leq; 100%  

### Part B.2 (20 points)

> **if** baseline latency is not **~2x** or **~4x** of **PartA** baseline &rarr; -5 points unless justified  
> Missing or incomplete information &rarr; -2 point for each question  
> Report values different from the ones achieved in **Part A.1**  &rarr; -3 points each   
> Insufficient description of technique(s) adopted &rarr; -2 points  

### Part C (Extra 20 points)

Awarded on a case-to-case to basis depending on how well the analysis/observations/implementations adhere to the questions stated.