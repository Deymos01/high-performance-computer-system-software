#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cassert>

const char* programSource = R"(
__kernel void hello_opencl(__global int* result, int num_threads) {
    int id = get_global_id(0);
    result[id] = id;

    barrier(CLK_GLOBAL_MEM_FENCE);
}
)";

int main() {
    cl_int err;
    cl_platform_id platform;

    err = clGetPlatformIDs(1, &platform, nullptr);
    assert(err == CL_SUCCESS);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    assert(err == CL_SUCCESS);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    assert(err == CL_SUCCESS);
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    assert(err == CL_SUCCESS);

    const int num_threads = 4;

    cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * num_threads, nullptr, &err);
    assert(err == CL_SUCCESS);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, nullptr, &err);
    assert(err == CL_SUCCESS);

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char buildLog[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, nullptr);
        std::cerr << "Build failed:\n" << buildLog << std::endl;
        return -1;
    }

    cl_kernel kernel = clCreateKernel(program, "hello_opencl", &err);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &resultBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &num_threads);
    assert(err == CL_SUCCESS);

    size_t globalSize = num_threads;
    size_t localSize = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
    assert(err == CL_SUCCESS);

    std::vector<int> results(num_threads);
    err = clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0, sizeof(int) * num_threads, results.data(), 0, nullptr, nullptr);
    assert(err == CL_SUCCESS);

    for (int i = 0; i < num_threads; ++i) {
        std::cout << "Hello from thread " << results[i] << std::endl;
    }

    clReleaseMemObject(resultBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
