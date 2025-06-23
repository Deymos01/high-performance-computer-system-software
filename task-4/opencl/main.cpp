#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

const char* kernelSource = R"CLC(
__kernel void matMul(
    __global float* A,
    __global float* B,
    __global float* C,
    int N) {

    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;

    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}
)CLC";

void generateMatrix(std::vector<float>& mat, int size) {
    for (int i = 0; i < size * size; ++i)
        mat[i] = static_cast<float>(rand() % 10);
}

int main() {
    std::vector<int> sizes = {10, 100, 1000, 2000};

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "matMul", &err);

    for (int size : sizes) {
        size_t bytes = size * size * sizeof(float);
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C(size * size, 0);

        generateMatrix(A, size);
        generateMatrix(B, size);

        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

        clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, A.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, B.data(), 0, nullptr, nullptr);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(kernel, 3, sizeof(int), &size);

        size_t globalSize[2] = {static_cast<size_t>(size), static_cast<size_t>(size)};

        auto start = std::chrono::high_resolution_clock::now();

        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);

        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Matrix size: " << size << "x" << size
                  << ", Execution time: " << elapsed.count() << " seconds" << std::endl;

        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
