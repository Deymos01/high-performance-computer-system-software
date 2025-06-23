#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>

const char* kernelSource = R"CLC(
__kernel void computeDerivativeX(__global const double* input,
                                 __global double* output,
                                 const int rows,
                                 const int cols,
                                 const double dx) {
    int gid = get_global_id(0);
    int j;

    if (gid >= rows) return;

    for (j = 0; j < cols; j++) {
        int idx = gid * cols + j;

        if (j == 0) {
            output[idx] = (input[gid * cols + j + 1] - input[idx]) / dx;
        } else if (j == cols - 1) {
            output[idx] = (input[idx] - input[gid * cols + j - 1]) / dx;
        } else {
            output[idx] = (input[gid * cols + j + 1] - input[gid * cols + j - 1]) / (2.0 * dx);
        }
    }
}
)CLC";

constexpr double dx = 0.01;

double func(double x, double y) {
    return x * (sin(x) + cos(y));
}

int main() {
    std::vector<int> sizes = {10, 100, 1000, 10000};

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to find an OpenCL platform." << std::endl;
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device." << std::endl;
        return 1;
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context." << std::endl;
        return 1;
    }

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program." << std::endl;
        return 1;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "Error in kernel build:\n" << buildLog.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "computeDerivativeX", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return 1;
    }

    for (int size : sizes) {
        int rows = size;
        int cols = size;
        size_t totalSize = rows * cols;

        std::vector<double> inputData(totalSize);
        std::vector<double> outputData(totalSize, 0);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                inputData[i * cols + j] = func(i * dx, j * dx);

        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(double) * totalSize, inputData.data(), &err);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             sizeof(double) * totalSize, nullptr, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create buffers." << std::endl;
            return 1;
        }

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
        clSetKernelArg(kernel, 2, sizeof(int), &rows);
        clSetKernelArg(kernel, 3, sizeof(int), &cols);
        clSetKernelArg(kernel, 4, sizeof(double), &dx);

        size_t globalWorkSize = rows;

        auto startTime = std::chrono::high_resolution_clock::now();

        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue kernel." << std::endl;
            return 1;
        }

        clFinish(queue);

        clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(double) * totalSize, outputData.data(), 0, nullptr, nullptr);

        auto endTime = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = endTime - startTime;

        std::cout << "Grid size: " << rows << "x" << cols
                  << ", Execution time: " << elapsed.count() << " seconds" << std::endl;

        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
