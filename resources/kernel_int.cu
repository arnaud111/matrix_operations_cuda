extern "C" __global__ void add(const int* matrix1, const int* matrix2, int* output) {
    output[threadIdx.x] = matrix1[threadIdx.x] + matrix2[threadIdx.x];
}

extern "C" __global__ void sub(const int* matrix1, const int* matrix2, int* output) {
    output[threadIdx.x] = matrix1[threadIdx.x] - matrix2[threadIdx.x];
}

extern "C" __global__ void dot(const int* matrix1, const int* matrix2, int* output) {
    output[threadIdx.x] = matrix1[threadIdx.x] * matrix2[threadIdx.x];
}

extern "C" __global__ void add_scalar(const int* matrix, int* output, int scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] + scalar;
}

extern "C" __global__ void sub_scalar(const int* matrix, int* output, int scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] - scalar;
}

extern "C" __global__ void mul_scalar(const int* matrix, int* output, int scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] * scalar;
}

extern "C" __global__ void div_scalar(const int* matrix, int* output, int scalar) {
    output[threadIdx.x] = matrix[threadIdx.x] / scalar;
}

extern "C" __global__ void scalar_sub(const int* matrix, int* output, int scalar) {
    output[threadIdx.x] = scalar - matrix[threadIdx.x];
}

extern "C" __global__ void scalar_div(const int* matrix, int* output, int scalar) {
    output[threadIdx.x] = scalar / matrix[threadIdx.x];
}