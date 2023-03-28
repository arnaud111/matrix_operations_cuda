//! This module contains functions to apply your own cuda function to matrices.
//!
//! # Usage
//!
//! ```
//! use cuda_driver_sys::*;
//! use matrix_operations_cuda::cuda_env::CudaEnv;
//! use matrix_operations_cuda::cuda_module::CudaModule;
//! use matrix_operations::{Matrix, matrix};
//! use matrix_operations_cuda::matrix_apply::apply_function_matrix;
//!
//! unsafe {
//!     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
//!
//!     let module = CudaModule::new(b"resources/kernel_test.ptx\0").unwrap();
//!     let function = module.load_function(b"mul_by_2\0").unwrap();
//!
//!     let matrix = matrix![[1.0f32, 2.00f32, 3.0f32],
//!                          [4.00f32, 5.00f32, 6.0f32]];
//!
//!     let result = apply_function_matrix(&matrix, &mut cuda_env, function).unwrap();
//!
//!     assert_eq!(result[0], [2.0f32, 4.0f32, 6.0f32]);
//!     assert_eq!(result[1], [8.0f32, 10.0f32, 12.0f32]);
//!
//!     module.free().unwrap();
//! }
//! ```
use std::error::Error;
use std::ffi::c_void;
use std::mem::size_of;
use cuda_driver_sys::CUfunction;
use matrix_operations::Matrix;
use crate::CudaEnv;

/// Runs a cuda function, as many times as there are values in the matrix, with a scalar as argument.
///
/// Signature of the function must be:
///
/// ```C
/// extern "C" __global__ void function_name(const float* matrix, float* output, float scalar)
/// ```
///
/// # Example
///
/// ```
/// use cuda_driver_sys::*;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
/// use matrix_operations_cuda::cuda_module::CudaModule;
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::matrix_apply::apply_function_matrix_scalar;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
///     let function = module.load_function(b"add_scalar\0").unwrap();
///
///     let matrix = matrix![[1.0, 2.0],
///                          [3.0, 4.0]];
///
///     let result = apply_function_matrix_scalar(&matrix, 2.0, &mut cuda_env, function).unwrap();
///
///     assert_eq!(result[0], [3.0, 4.0]);
///     assert_eq!(result[1], [5.0, 6.0]);
///
///     module.free().unwrap();
/// }
/// ```
pub unsafe fn apply_function_matrix_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv, function: CUfunction) -> Result<Matrix<f32>, Box<dyn Error>> {
    let matrix_data = matrix.as_slice();

    let max_threads_per_block = cuda_env.get_max_threads_per_block();
    let max_blocks_per_grid = cuda_env.get_max_block_per_grid();
    let (block_dim, grid_dim) = CudaEnv::get_block_and_grid_dim(matrix_data.len(), max_threads_per_block, max_blocks_per_grid);

    let matrix_data_ptr = cuda_env.allocate(matrix_data.len() * size_of::<f32>()).unwrap();
    cuda_env.copy_host_to_device(matrix_data_ptr, matrix_data).unwrap();

    let device_ptr_out = cuda_env.allocate(matrix_data.len() * size_of::<f32>()).unwrap();
    cuda_env.set_empty_device_data(device_ptr_out, matrix_data.len() * size_of::<f32>()).unwrap();

    let args = [
        &matrix_data_ptr as *const _ as *mut c_void,
        &device_ptr_out as *const _ as *mut c_void,
        &scalar as *const _ as *mut c_void,
    ];
    cuda_env.launch(function, &args, grid_dim, block_dim).unwrap();

    let mut result = vec![0.0f32; matrix_data.len()];
    cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();

    cuda_env.free_data(matrix_data_ptr)?;
    cuda_env.free_data(device_ptr_out)?;

    Matrix::from_slice(result.as_slice(), matrix.shape())
}

/// Runs a cuda function, as many times as there are values in the matrix.
///
/// Signature of the function must be:
///
/// ```C
/// extern "C" __global__ void function_name(const float* matrix1, const float* matrix2, float* output)
/// ```
///
/// # Example
///
/// ```
/// use cuda_driver_sys::*;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
/// use matrix_operations_cuda::cuda_module::CudaModule;
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::matrix_apply::apply_function_two_matrices;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
///     let function = module.load_function(b"add\0").unwrap();
///
///     let matrix1 = matrix![[1.0, 2.0],
///                           [3.0, 4.0]];
///     let matrix2 = matrix![[1.0, 2.0],
///                           [3.0, 4.0]];
///
///     let result = apply_function_two_matrices(&matrix1, &matrix2, &mut cuda_env, function).unwrap();
///
///     assert_eq!(result[0], [2.0, 4.0]);
///     assert_eq!(result[1], [6.0, 8.0]);
///
///     module.free().unwrap();
/// }
/// ```
pub unsafe fn apply_function_two_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &mut CudaEnv, function: CUfunction) -> Result<Matrix<f32>, Box<dyn Error>> {
    let matrix1_data = matrix1.as_slice();
    let matrix2_data = matrix2.as_slice();

    let max_threads_per_block = cuda_env.get_max_threads_per_block();
    let max_blocks_per_grid = cuda_env.get_max_block_per_grid();
    let (block_dim, grid_dim) = CudaEnv::get_block_and_grid_dim(matrix1_data.len(), max_threads_per_block, max_blocks_per_grid);

    let matrix1_data_ptr = cuda_env.allocate(matrix1_data.len() * size_of::<f32>()).unwrap();
    cuda_env.copy_host_to_device(matrix1_data_ptr, matrix1_data).unwrap();

    let matrix2_data_ptr = cuda_env.allocate(matrix2_data.len() * size_of::<f32>()).unwrap();
    cuda_env.copy_host_to_device(matrix2_data_ptr, matrix2_data).unwrap();

    let device_ptr_out = cuda_env.allocate(matrix1_data.len() * size_of::<f32>()).unwrap();
    cuda_env.set_empty_device_data(device_ptr_out, matrix1_data.len() * size_of::<f32>()).unwrap();

    let args = [
        &matrix1_data_ptr as *const _ as *mut c_void,
        &matrix2_data_ptr as *const _ as *mut c_void,
        &device_ptr_out as *const _ as *mut c_void,
    ];
    cuda_env.launch(function, &args, grid_dim, block_dim).unwrap();

    let mut result = vec![0.0f32; matrix1_data.len()];
    cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();

    cuda_env.free_data(matrix1_data_ptr)?;
    cuda_env.free_data(matrix2_data_ptr)?;
    cuda_env.free_data(device_ptr_out)?;

    Matrix::from_slice(result.as_slice(), matrix1.shape())
}

/// Runs a cuda function a number of times equal to the returned_shape.
///
/// Signature of the function must be:
/// ```C
/// extern "C" __global__ void function_name(const float* matrix1, const float* matrix2, float* output, int matrix1_row, int matrix1_col, int matrix2_row, int matrix2_col)
/// ```
///
/// # Example
///
/// ```
/// use cuda_driver_sys::*;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
/// use matrix_operations_cuda::cuda_module::CudaModule;
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::matrix_apply::apply_function_two_matrices_with_shapes;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
///     let function = module.load_function(b"dot\0").unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32, 3.0f32],
///                           [4.00f32, 5.00f32, 6.0f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32],
///                           [14.00f32, 15.00f32]];
///
///     let result = apply_function_two_matrices_with_shapes(&matrix1, &matrix2, (2, 2), &mut cuda_env, function).unwrap();
///
///     assert_eq!(result[0], [76.0f32, 82.0f32]);
///     assert_eq!(result[1], [184.0f32, 199.0f32]);
///
///     module.free().unwrap();
/// }
/// ```
pub unsafe fn apply_function_two_matrices_with_shapes(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, returned_shape: (usize, usize), cuda_env: &mut CudaEnv, function: CUfunction) -> Result<Matrix<f32>, Box<dyn Error>> {
    let matrix1_data = matrix1.as_slice();
    let matrix2_data = matrix2.as_slice();

    let max_threads_per_block = cuda_env.get_max_threads_per_block();
    let max_blocks_per_grid = cuda_env.get_max_block_per_grid();
    let (block_dim, grid_dim) = CudaEnv::get_block_and_grid_dim(returned_shape.0 * returned_shape.1, max_threads_per_block, max_blocks_per_grid);

    let matrix1_data_ptr = cuda_env.allocate(matrix1_data.len() * size_of::<f32>()).unwrap();
    cuda_env.copy_host_to_device(matrix1_data_ptr, matrix1_data).unwrap();

    let matrix2_data_ptr = cuda_env.allocate(matrix2_data.len() * size_of::<f32>()).unwrap();
    cuda_env.copy_host_to_device(matrix2_data_ptr, matrix2_data).unwrap();

    let device_ptr_out = cuda_env.allocate(returned_shape.0 * returned_shape.1 * size_of::<f32>()).unwrap();
    cuda_env.set_empty_device_data(device_ptr_out, returned_shape.0 * returned_shape.1 * size_of::<f32>()).unwrap();

    let args = [
        &matrix1_data_ptr as *const _ as *mut c_void,
        &matrix2_data_ptr as *const _ as *mut c_void,
        &device_ptr_out as *const _ as *mut c_void,
        &matrix1.shape().0 as *const _ as *mut c_void,
        &matrix1.shape().1 as *const _ as *mut c_void,
        &matrix2.shape().0 as *const _ as *mut c_void,
        &matrix2.shape().1 as *const _ as *mut c_void,
    ];
    cuda_env.launch(function, &args, grid_dim, block_dim).unwrap();

    let mut result = vec![0.0f32; returned_shape.0 * returned_shape.1];
    cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();

    cuda_env.free_data(matrix1_data_ptr)?;
    cuda_env.free_data(matrix2_data_ptr)?;
    cuda_env.free_data(device_ptr_out)?;

    Matrix::from_slice(result.as_slice(), returned_shape)
}

/// Runs a cuda function, as many times as there are values in the matrix
///
/// Signature of the function must be:
///
/// ```C
/// extern "C" __global__ void function_name(const float* matrix, float* output)
/// ```
///
/// # Example
///
/// ```
/// use cuda_driver_sys::*;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
/// use matrix_operations_cuda::cuda_module::CudaModule;
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::matrix_apply::apply_function_matrix;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let module = CudaModule::new(b"resources/kernel_test.ptx\0").unwrap();
///     let function = module.load_function(b"mul_by_2\0").unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32, 3.0f32],
///                          [4.00f32, 5.00f32, 6.0f32]];
///
///     let result = apply_function_matrix(&matrix, &mut cuda_env, function).unwrap();
///
///     assert_eq!(result[0], [2.0f32, 4.0f32, 6.0f32]);
///     assert_eq!(result[1], [8.0f32, 10.0f32, 12.0f32]);
///
///     module.free().unwrap();
/// }
/// ```
pub unsafe fn apply_function_matrix(matrix: &Matrix<f32>, cuda_env: &mut CudaEnv, function: CUfunction) -> Result<Matrix<f32>, Box<dyn Error>> {
    let matrix_data = matrix.as_slice();

    let max_threads_per_block = cuda_env.get_max_threads_per_block();
    let max_blocks_per_grid = cuda_env.get_max_block_per_grid();
    let (block_dim, grid_dim) = CudaEnv::get_block_and_grid_dim(matrix_data.len(), max_threads_per_block, max_blocks_per_grid);

    let matrix_data_ptr = cuda_env.allocate(matrix_data.len() * size_of::<f32>()).unwrap();
    cuda_env.copy_host_to_device(matrix_data_ptr, matrix_data).unwrap();

    let device_ptr_out = cuda_env.allocate(matrix_data.len() * size_of::<f32>()).unwrap();
    cuda_env.set_empty_device_data(device_ptr_out, matrix_data.len() * size_of::<f32>()).unwrap();

    let args = [
        &matrix_data_ptr as *const _ as *mut c_void,
        &device_ptr_out as *const _ as *mut c_void,
    ];
    cuda_env.launch(function, &args, grid_dim, block_dim).unwrap();

    let mut result = vec![0.0f32; matrix_data.len()];
    cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();

    cuda_env.free_data(matrix_data_ptr)?;
    cuda_env.free_data(device_ptr_out)?;

    Matrix::from_slice(result.as_slice(), matrix.shape())
}

/// Runs a cuda function a number of times equal to the returned_shape. with the matrix and is shape as parameters
///
/// Signature of the function must be:
/// ```C
/// extern "C" __global__ void function_name(const float* matrix, float* output, int matrix_row, int matrix_col)
/// ```
///
/// # Example
///
/// ```
/// use cuda_driver_sys::*;
/// use matrix_operations_cuda::cuda_env::CudaEnv;
/// use matrix_operations_cuda::cuda_module::CudaModule;
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::matrix_apply::apply_function_matrix_with_shapes;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///
///     let module = CudaModule::new(b"resources/kernel_test.ptx\0").unwrap();
///     let function = module.load_function(b"sum_column\0").unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32, 3.0f32],
///                          [4.00f32, 5.00f32, 6.0f32]];
///
///     let result = apply_function_matrix_with_shapes(&matrix, (1, 3), &mut cuda_env, function).unwrap();
///
///     assert_eq!(result[0], [5.0f32, 7.0f32, 9.0f32]);
///
///     module.free().unwrap();
/// }
/// ```
pub unsafe fn apply_function_matrix_with_shapes(matrix: &Matrix<f32>, returned_shape: (usize, usize), cuda_env: &mut CudaEnv, function: CUfunction) -> Result<Matrix<f32>, Box<dyn Error>> {
    let matrix_data = matrix.as_slice();

    let max_threads_per_block = cuda_env.get_max_threads_per_block();
    let max_blocks_per_grid = cuda_env.get_max_block_per_grid();
    let (block_dim, grid_dim) = CudaEnv::get_block_and_grid_dim(matrix_data.len(), max_threads_per_block, max_blocks_per_grid);

    let matrix_data_ptr = cuda_env.allocate(matrix_data.len() * size_of::<f32>()).unwrap();
    cuda_env.copy_host_to_device(matrix_data_ptr, matrix_data).unwrap();

    let device_ptr_out = cuda_env.allocate(returned_shape.0 * returned_shape.1 * size_of::<f32>()).unwrap();
    cuda_env.set_empty_device_data(device_ptr_out, returned_shape.0 * returned_shape.1 * size_of::<f32>()).unwrap();

    let args = [
        &matrix_data_ptr as *const _ as *mut c_void,
        &device_ptr_out as *const _ as *mut c_void,
        &matrix.shape().0 as *const _ as *mut c_void,
        &matrix.shape().1 as *const _ as *mut c_void,
    ];
    cuda_env.launch(function, &args, grid_dim, block_dim).unwrap();

    let mut result = vec![0.0f32; returned_shape.0 * returned_shape.1];
    cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();

    cuda_env.free_data(matrix_data_ptr)?;
    cuda_env.free_data(device_ptr_out)?;

    Matrix::from_slice(result.as_slice(), returned_shape)
}
