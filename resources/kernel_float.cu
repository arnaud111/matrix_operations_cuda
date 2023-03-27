extern "C" __global__ void add(const float* matrix1, const float* matrix2, float* output) {
    output[threadIdx.x] = matrix1[threadIdx.x] + matrix2[threadIdx.x];
}

extern "C" __global__ void sub(const float* matrix1, const float* matrix2, float* output) {
    output[threadIdx.x] = matrix1[threadIdx.x] - matrix2[threadIdx.x];
}

extern "C" __global__ void dot(const float* matrix1, const float* matrix2, float* output) {
    output[threadIdx.x] = matrix1[threadIdx.x] * matrix2[threadIdx.x];
}

extern "C" __global__ void add_scalar(const float* matrix, float* output, float scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] + scalar;
}

extern "C" __global__ void sub_scalar(const float* matrix, float* output, float scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] - scalar;
}

extern "C" __global__ void mul_scalar(const float* matrix, float* output, float scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] * scalar;
}

extern "C" __global__ void div_scalar(const float* matrix, float* output, float scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] / scalar;
}

extern "C" __global__ void scalar_sub(const float* matrix, float* output, float scalar) {
    output[threadIdx.x] = scalar - matrix[threadIdx.x];
}

extern "C" __global__ void scalar_div(const float* matrix, float* output, float scalar) {
    output[threadIdx.x] = scalar / matrix[threadIdx.x];
}