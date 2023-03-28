extern "C" __global__ void mul_by_2(const float* matrix, float* output) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    output[i] = matrix[i] * 2;
}

extern "C" __global__ void sum_column(const float* matrix, float* output, int matrix_row, int matrix_col) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    int sum = 0.0f;
    for (int j = 0; j < matrix_row; j++) {
        sum += matrix[j * matrix_col + i];
    }
    output[i] = sum;
}
