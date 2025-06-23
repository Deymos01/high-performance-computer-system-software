#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

const char* kernelSource = R"CLC(
__kernel void reduce_sum(__global const int* input, __global int* partialSums, const int N) {
    int gid = get_global_id(0);
    int groupSize = get_local_size(0);
    int lid = get_local_id(0);
    __local int localSums[256];

    int sum = 0;
    for (int i = gid; i < N; i += get_global_size(0)) {
        sum += input[i];
    }
    localSums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = groupSize / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            localSums[lid] += localSums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partialSums[get_group_id(0)] = localSums[0];
    }
}
)CLC";

void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error (" << err << "): " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    std::vector<int> sizes = {10, 1000, 10000000};
    cl_int err;

    cl_platform_id platform;
    cl_device_id device;
    check(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr), "clGetDeviceIDs");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    check(err, "clCreateCommandQueue");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build error:\n" << log << std::endl;
        std::exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "reduce_sum", &err);
    check(err, "clCreateKernel");

    for (int n : sizes) {
        std::vector<int> input(n);
        for (int i = 0; i < n; ++i) input[i] = rand() % 10;

        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, input.data(), &err);
        check(err, "clCreateBuffer input");

        const int localSize = 256;
        int numGroups = (n + localSize - 1) / localSize;
        cl_mem partialBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * numGroups, nullptr, &err);
        check(err, "clCreateBuffer partial");

        check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "set arg 0");
        check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &partialBuffer), "set arg 1");
        check(clSetKernelArg(kernel, 2, sizeof(int), &n), "set arg 2");

        size_t globalSize = localSize * numGroups;
        size_t local = localSize;

        auto start = std::chrono::high_resolution_clock::now();

        check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &local, 0, nullptr, nullptr), "enqueue kernel");
        std::vector<int> partialSums(numGroups);
        check(clEnqueueReadBuffer(queue, partialBuffer, CL_TRUE, 0, sizeof(int) * numGroups, partialSums.data(), 0, nullptr, nullptr), "read partial");

        int finalSum = 0;
        for (int s : partialSums) finalSum += s;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Array size: " << n
                  << ", Sum: " << finalSum
                  << ", Time: " << duration.count() << " s" << std::endl;

        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(partialBuffer);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}