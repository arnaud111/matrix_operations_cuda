use std::error::Error;
use matrix_operations::Matrix;
use crate::matrix_apply::{apply_function_matrix_scalar, apply_function_two_matrices, apply_function_two_matrices_with_shapes};
pub use crate::cuda_env::CudaEnv;
pub use crate::cuda_module::CudaModule;

pub mod cuda_module;
pub mod cuda_env;
pub mod matrix_apply;

/// Add a scalar to a matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::add_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = add_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [3.00f32, 4.00f32]);
///     assert_eq!(result[1], [5.00f32, 6.00f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::add_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = add_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 + 2.0);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn add_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"add_scalar\0")?;
    let result = apply_function_matrix_scalar(matrix, scalar, cuda_env, function);
    module.free()?;
    result
}

/// Subtract a scalar to a matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::sub_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = sub_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [-1.00f32, 0.00f32]);
///     assert_eq!(result[1], [1.00f32, 2.00f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::sub_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = sub_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 - 2.0);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn sub_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"sub_scalar\0")?;
    let result = apply_function_matrix_scalar(matrix, scalar, cuda_env, function);
    module.free()?;
    result
}

/// Multiply a scalar to a matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::mul_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = mul_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [2.00f32, 4.00f32]);
///     assert_eq!(result[1], [6.00f32, 8.00f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::mul_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = mul_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 * 2.0);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn mul_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"mul_scalar\0")?;
    let result = apply_function_matrix_scalar(matrix, scalar, cuda_env, function);
    module.free()?;
    result
}

/// Divide a scalar to a matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::div_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = div_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [0.50f32, 1.00f32]);
///     assert_eq!(result[1], [1.50f32, 2.00f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::div_scalar;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = div_scalar(&matrix, 2.0, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 / 2.0);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn div_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"div_scalar\0")?;
    let result= apply_function_matrix_scalar(matrix, scalar, cuda_env, function);
    module.free()?;
    result
}

/// Subtract scalar - matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::scalar_sub;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = scalar_sub(&matrix, 2.0, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [1.00f32, 0.00f32]);
///     assert_eq!(result[1], [-1.00f32, -2.00f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::scalar_sub;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = scalar_sub(&matrix, 2.0, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 - i as f32);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn scalar_sub(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"scalar_sub\0")?;
    let result = apply_function_matrix_scalar(matrix, scalar, cuda_env, function);
    module.free()?;
    result
}

/// Divide scalar / matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::scalar_div;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = scalar_div(&matrix, 2.0, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [2.00f32, 1.00f32]);
///     assert_eq!(result[1], [0.6666667f32, 0.5f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::scalar_div;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = scalar_div(&matrix, 2.0, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 / i as f32);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn scalar_div(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"scalar_div\0")?;
    let result = apply_function_matrix_scalar(matrix, scalar, cuda_env, function);
    module.free()?;
    result
}

/// Add two matrices
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::add_matrices;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32]];
///
///     let result = add_matrices(&matrix1, &matrix2, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [11.00f32, 13.00f32]);
///     assert_eq!(result[1], [15.00f32, 17.00f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::add_matrices;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix1 = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = add_matrices(&matrix1, &matrix1, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 * i as f32);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// # Errors
///
/// If matrices have different sizes
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::add_matrices;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32],
///                           [14.00f32, 15.00f32]];
///
///     let result = add_matrices(&matrix1, &matrix2, &cuda_env);
///
///     assert!(result.is_err());
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn add_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {
    if matrix1.shape().0 != matrix2.shape().0 || matrix1.shape().1 != matrix2.shape().1 {
        return Err("Matrices must have the same size".into());
    }

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"add\0")?;
    let result = apply_function_two_matrices(matrix1, matrix2, cuda_env, function);
    module.free()?;
    result
}

/// Subtract two matrices
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::sub_matrices;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32]];
///
///     let result = sub_matrices(&matrix1, &matrix2, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [-9.0f32, -9.00f32]);
///     assert_eq!(result[1], [-9.00f32, -9.00f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::sub_matrices;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix1 = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = sub_matrices(&matrix1, &matrix1, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 0.0f32);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// # Errors
///
/// If the matrices have different sizes
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::sub_matrices;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///    let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32],
///                           [14.00f32, 15.00f32]];
///
///     let result = sub_matrices(&matrix1, &matrix2, &cuda_env);
///
///     assert!(result.is_err());
///
///     cuda_env.free().unwrap();
/// }
pub unsafe fn sub_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {
    if matrix1.shape().0 != matrix2.shape().0 || matrix1.shape().1 != matrix2.shape().1 {
        return Err("Matrices must have the same size".into());
    }

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"sub\0")?;
    let result = apply_function_two_matrices(matrix1, matrix2, cuda_env, function);
    module.free()?;
    result
}

/// Multiply two matrices
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::dot;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32, 3.0f32],
///                           [4.00f32, 5.00f32, 6.0f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32],
///                           [14.00f32, 15.00f32]];
///
///     let result = dot(&matrix1, &matrix2, &cuda_env).unwrap();
///
///     assert_eq!(result[0], [76.0f32, 82.0f32]);
///     assert_eq!(result[1], [184.0f32, 199.0f32]);
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::dot;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let mut data1 = vec![1.0f32; 1000000];
///     let mut data2 = vec![2.0f32; 1000000];
///
///     let matrix1 = Matrix::new(data1.clone(), (1000, 1000)).unwrap();
///     let matrix2 = Matrix::new(data2.clone(), (1000, 1000)).unwrap();
///
///     let result = dot(&matrix1, &matrix2, &cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data1.len() {
///         assert_eq!(data_result[i], 2.0f32 * 1000.0f32);
///     }
///
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// # Errors
///
/// If the number of columns of the first matrix is not equal to the number of rows of the second matrix, an error is returned.
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::dot;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32, 3.0f32],
///                           [4.00f32, 5.00f32, 6.0f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32]];
///
///     let result = dot(&matrix1, &matrix2, &cuda_env);
///
///     assert!(result.is_err());
///
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn dot(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {
    if matrix1.shape().1 != matrix2.shape().0 {
        return Err("The number of columns of the first matrix must be equal to the number of rows of the second matrix".into());
    }

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"dot\0")?;
    let result = apply_function_two_matrices_with_shapes(matrix1, matrix2, (matrix1.shape().0, matrix2.shape().1), cuda_env, function);
    module.free()?;
    result
}
