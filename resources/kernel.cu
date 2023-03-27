extern "C" __global__ void add(const float* input, float* output) {
    output[threadIdx.x] = input[threadIdx.x] + 1;
}