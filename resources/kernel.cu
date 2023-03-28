extern "C" __global__ void add(const float* matrix1, const float* matrix2, float* output) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix1[i] + matrix2[i];
}

extern "C" __global__ void sub(const float* matrix1, const float* matrix2, float* output) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix1[i] - matrix2[i];
}

extern "C" __global__ void dot(const float* matrix1, const float* matrix2, float* output, int matrix1_row, int matrix1_col, int matrix2_row, int matrix2_col) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    int col = i % matrix2_col;
    int row = i / matrix2_col;
    float sum = 0.0f;
    for (int j = 0; j < matrix2_row; j++) {
        sum += matrix1[row * matrix1_col + j] * matrix2[j * matrix2_col + col];
    }
    output[i] = sum;
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