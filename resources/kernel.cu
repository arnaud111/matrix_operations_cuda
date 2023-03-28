extern "C" __global__ void add(const float* matrix1, const float* matrix2, float* output) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix1[i] + matrix2[i];
}

extern "C" __global__ void sub(const float* matrix1, const float* matrix2, float* output) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix1[i] - matrix2[i];
}

extern "C" __global__ void dot(const float* matrix1, const float* matrix2, float* output) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix1[i] * matrix2[i];
}

extern "C" __global__ void add_scalar(const float* matrix, float* output, float scalar) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix[i] + scalar;
}

extern "C" __global__ void sub_scalar(const float* matrix, float* output, float scalar) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix[i] - scalar;
}

extern "C" __global__ void mul_scalar(const float* matrix, float* output, float scalar) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix[i] * scalar;
}

extern "C" __global__ void div_scalar(const float* matrix, float* output, float scalar) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix[i] / scalar;
}

extern "C" __global__ void scalar_sub(const float* matrix, float* output, float scalar) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = scalar - matrix[i];
}

extern "C" __global__ void scalar_div(const float* matrix, float* output, float scalar) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = scalar / matrix[i];
}