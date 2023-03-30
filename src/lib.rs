//! # Matrix_Operations_Cuda
//!
//! Matrix_Operations_Cuda is a Rust crate for performing matrix operations using CUDA
//!
//! ## Installation
//!
//! - ### Matrix_Operations_Cuda
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! matrix_operations_cuda = "0.1.0"
//! ```
//!
//! - ### Matrix_Operations
//!
//! To works with `matrix_operations_cuda`, you need to install the core matrix_operations from `matrix_operations`
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! matrix_operations = "0.1.3"
//! ```
//!
//! - ### CUDA
//! **This crate does NOT include CUDA itself. You need to install on your own.**
//! [the official installer](https://developer.nvidia.com/cuda-downloads)
//!
//! ## Usage
//!
//! This crate allow to use common operations using cuda:
//!
//! ```
//! use matrix_operations::matrix;
//! use matrix_operations_cuda::{add_matrices, add_scalar, CudaEnv, CudaModule, dot, sub_matrices};
//!
//! println!("test");
//! let cuda_env;
//! unsafe {
//!     cuda_env = CudaEnv::new(0, 0).unwrap();
//! }
//! println!("test");
//! let module;
//! unsafe {
//!     module = CudaModule::default().unwrap();
//! }
//! println!("test");
//!
//! let m1 = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let m2 = matrix![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//!
//! println!("test");
//! let m3;
//! unsafe {
//!     m3 = dot(&m1, &m2, &cuda_env, &module).unwrap();
//! }
//! println!("test");
//!
//! assert_eq!(m3[0], [22.0, 28.0]);
//! assert_eq!(m3[1], [49.0, 64.0]);
//!
//! let m4;
//! unsafe {
//!     m4 = add_scalar(&m3, 10.0, &cuda_env, &module).unwrap();
//! }
//!
//! println!("test");
//! assert_eq!(m4[0], [32.0, 38.0]);
//! assert_eq!(m4[1], [59.0, 74.0]);
//!
//! let m5;
//! unsafe {
//!     m5 = sub_matrices(&m4, &m3, &cuda_env, &module).unwrap();
//! }
//!
//! println!("test");
//! assert_eq!(m5[0], [10.0, 10.0]);
//! assert_eq!(m5[1], [10.0, 10.0]);
//!
//! unsafe {
//!     module.free().unwrap();
//! }
//! ```
//!
//! You also can import your own module from a `.ptx` file or from a module data as `Vec<u8>`
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
//!     let matrix = matrix![[1.0, 2.0, 3.0],
//!                          [4.0, 5.0, 6.0]];
//!
//!     let result = apply_function_matrix(&matrix, &cuda_env, function).unwrap();
//!
//!     assert_eq!(result[0], [2.0, 4.0, 6.0]);
//!     assert_eq!(result[1], [8.0, 10.0, 12.0]);
//!
//!     module.free().unwrap();
//! }
//! ```
//!
//! ## Features
//!
//! - Initialize a cuda environment
//! - Launch common operations on matrices
//! - Import and use custom kernel to perform custom operations on matrices
//! - Allocate and Free memory in GPU
//! - Copy data between Host and Device

extern crate matrix_operations;

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
/// use matrix_operations_cuda::{add_scalar, CudaModule};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = add_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [3.00f32, 4.00f32]);
///     assert_eq!(result[1], [5.00f32, 6.00f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{add_scalar, CudaModule};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = add_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 + 2.0);
///     }
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn add_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {

    let function = module.load_function(b"add_scalar\0")?;
    apply_function_matrix_scalar(matrix, scalar, cuda_env, function)
}

/// Subtract a scalar to a matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, sub_scalar};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = sub_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [-1.00f32, 0.00f32]);
///     assert_eq!(result[1], [1.00f32, 2.00f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, sub_scalar};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = sub_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 - 2.0);
///     }
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn sub_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {

    let function = module.load_function(b"sub_scalar\0")?;
    apply_function_matrix_scalar(matrix, scalar, cuda_env, function)
}

/// Multiply a scalar to a matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, mul_scalar};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = mul_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [2.00f32, 4.00f32]);
///     assert_eq!(result[1], [6.00f32, 8.00f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, mul_scalar};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = mul_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 * 2.0);
///     }
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn mul_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {

    let function = module.load_function(b"mul_scalar\0")?;
    apply_function_matrix_scalar(matrix, scalar, cuda_env, function)
}

/// Divide a scalar to a matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, div_scalar};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = div_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [0.50f32, 1.00f32]);
///     assert_eq!(result[1], [1.50f32, 2.00f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, div_scalar};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = div_scalar(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 / 2.0);
///     }
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn div_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {

    let function = module.load_function(b"div_scalar\0")?;
    apply_function_matrix_scalar(matrix, scalar, cuda_env, function)
}

/// Subtract scalar - matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, scalar_sub};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = scalar_sub(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [1.00f32, 0.00f32]);
///     assert_eq!(result[1], [-1.00f32, -2.00f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, scalar_sub};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = scalar_sub(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 - i as f32);
///     }
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn scalar_sub(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {

    let function = module.load_function(b"scalar_sub\0")?;
    apply_function_matrix_scalar(matrix, scalar, cuda_env, function)
}

/// Divide scalar / matrix
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, scalar_div};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix = matrix![[1.0f32, 2.00f32],
///                          [3.00f32, 4.00f32]];
///
///     let result = scalar_div(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [2.00f32, 1.00f32]);
///     assert_eq!(result[1], [0.6666667f32, 0.5f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, scalar_div};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = scalar_div(&matrix, 2.0, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 / i as f32);
///     }
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn scalar_div(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {

    let function = module.load_function(b"scalar_div\0")?;
    apply_function_matrix_scalar(matrix, scalar, cuda_env, function)
}

/// Add two matrices
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{add_matrices, CudaModule};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32]];
///
///     let result = add_matrices(&matrix1, &matrix2, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [11.00f32, 13.00f32]);
///     assert_eq!(result[1], [15.00f32, 17.00f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{add_matrices, CudaModule};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix1 = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = add_matrices(&matrix1, &matrix1, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 * i as f32);
///     }
///
///     module.free().unwrap();
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
/// use matrix_operations_cuda::{add_matrices, CudaModule};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32],
///                           [14.00f32, 15.00f32]];
///
///     let result = add_matrices(&matrix1, &matrix2, &cuda_env, &module);
///
///     assert!(result.is_err());
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn add_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {
    if matrix1.shape().0 != matrix2.shape().0 || matrix1.shape().1 != matrix2.shape().1 {
        return Err("Matrices must have the same size".into());
    }

    let function = module.load_function(b"add\0")?;
    apply_function_two_matrices(matrix1, matrix2, cuda_env, function)
}

/// Subtract two matrices
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, sub_matrices};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32]];
///
///     let result = sub_matrices(&matrix1, &matrix2, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [-9.0f32, -9.00f32]);
///     assert_eq!(result[1], [-9.00f32, -9.00f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, sub_matrices};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data = vec![0.0f32; 1000000];
///     for i in 0..data.len() {
///         data[i] = i as f32;
///     }
///
///     let matrix1 = Matrix::new(data.clone(), (1000, 1000)).unwrap();
///
///     let result = sub_matrices(&matrix1, &matrix1, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 0.0f32);
///     }
///
///     module.free().unwrap();
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
/// use matrix_operations_cuda::{CudaModule, sub_matrices};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///    let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32],
///                           [3.00f32, 4.00f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32],
///                           [14.00f32, 15.00f32]];
///
///     let result = sub_matrices(&matrix1, &matrix2, &cuda_env, &module);
///
///     assert!(result.is_err());
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
pub unsafe fn sub_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {
    if matrix1.shape().0 != matrix2.shape().0 || matrix1.shape().1 != matrix2.shape().1 {
        return Err("Matrices must have the same size".into());
    }

    let function = module.load_function(b"sub\0")?;
    apply_function_two_matrices(matrix1, matrix2, cuda_env, function)
}

/// Multiply two matrices
///
/// # Example
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, dot};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32, 3.0f32],
///                           [4.00f32, 5.00f32, 6.0f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32],
///                           [14.00f32, 15.00f32]];
///
///     let result = dot(&matrix1, &matrix2, &cuda_env, &module).unwrap();
///
///     assert_eq!(result[0], [76.0f32, 82.0f32]);
///     assert_eq!(result[1], [184.0f32, 199.0f32]);
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
///
/// Works with big matrices:
///
/// ```
/// use matrix_operations::{Matrix, matrix};
/// use matrix_operations_cuda::{CudaModule, dot};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let mut data1 = vec![1.0f32; 1000000];
///     let mut data2 = vec![2.0f32; 1000000];
///
///     let matrix1 = Matrix::new(data1.clone(), (1000, 1000)).unwrap();
///     let matrix2 = Matrix::new(data2.clone(), (1000, 1000)).unwrap();
///
///     let result = dot(&matrix1, &matrix2, &cuda_env, &module).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data1.len() {
///         assert_eq!(data_result[i], 2.0f32 * 1000.0f32);
///     }
///
///     module.free().unwrap();
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
/// use matrix_operations_cuda::{CudaModule, dot};
/// use matrix_operations_cuda::cuda_env::CudaEnv;
///
/// unsafe {
///     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
///     let module = CudaModule::default().unwrap();
///
///     let matrix1 = matrix![[1.0f32, 2.00f32, 3.0f32],
///                           [4.00f32, 5.00f32, 6.0f32]];
///
///     let matrix2 = matrix![[10.0f32, 11.00f32],
///                           [12.00f32, 13.00f32]];
///
///     let result = dot(&matrix1, &matrix2, &cuda_env, &module);
///
///     assert!(result.is_err());
///
///     module.free().unwrap();
///     cuda_env.free().unwrap();
/// }
/// ```
pub unsafe fn dot(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &CudaEnv, module: &CudaModule) -> Result<Matrix<f32>, Box<dyn Error>> {
    if matrix1.shape().1 != matrix2.shape().0 {
        return Err("The number of columns of the first matrix must be equal to the number of rows of the second matrix".into());
    }

    let function = module.load_function(b"dot\0")?;
    apply_function_two_matrices_with_shapes(matrix1, matrix2, (matrix1.shape().0, matrix2.shape().1), cuda_env, function)
}
