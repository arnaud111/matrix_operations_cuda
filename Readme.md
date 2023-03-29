 # Matrix_Operations_Cuda
[![Crates.io](https://img.shields.io/crates/v/matrix_operations.svg)](https://crates.io/crates/matrix_operations_cuda)
[![Docs.rs](https://docs.rs/matrix_operations/badge.svg)](https://docs.rs/matrix_operations_cuda)

 Matrix_Operations_Cuda is a Rust crate for performing matrix operations using CUDA

 ## Installation

 - ### Matrix_Operations_Cuda

 Add the following to your `Cargo.toml` file:

 ```toml
 [dependencies]
 matrix_operations_cuda = "0.1.0"
 ```

 - ### Matrix_Operations

 To works with `matrix_operations_cuda`, you need to install the core matrix_operations from `matrix_operations`

 Add the following to your `Cargo.toml` file:

 ```toml
 [dependencies]
 matrix_operations = "0.1.3"
 ```

 - ### CUDA
 **This crate does NOT include CUDA itself. You need to install on your own.**
 [the official installer](https://developer.nvidia.com/cuda-downloads)

 ## Usage

 This crate allow to use common operations using cuda:

 ```rust
 use matrix_operations::matrix;
 use matrix_operations_cuda::{add_matrices, add_scalar, CudaEnv, dot, sub_matrices};

 let cuda_env;
 unsafe {
     cuda_env = CudaEnv::new(0, 0).unwrap();
 }

 let m1 = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
 let m2 = matrix![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

 let m3;
 unsafe {
     m3 = dot(&m1, &m2, &cuda_env).unwrap();
 }

 assert_eq!(m3[0], [22.0, 28.0]);
 assert_eq!(m3[1], [49.0, 64.0]);

 let m4;
 unsafe {
     m4 = add_scalar(&m3, 10.0, &cuda_env).unwrap();
 }

 assert_eq!(m4[0], [32.0, 38.0]);
 assert_eq!(m4[1], [59.0, 74.0]);

 let m5;
 unsafe {
     m5 = sub_matrices(&m4, &m3, &cuda_env).unwrap();
 }

 assert_eq!(m5[0], [10.0, 10.0]);
 assert_eq!(m5[1], [10.0, 10.0]);
 ```

 You also can import your own module from a `.ptx` file or from a module data as `Vec<u8>`

 ```rust
 use cuda_driver_sys::*;
 use matrix_operations_cuda::cuda_env::CudaEnv;
 use matrix_operations_cuda::cuda_module::CudaModule;
 use matrix_operations::{Matrix, matrix};
 use matrix_operations_cuda::matrix_apply::apply_function_matrix;

 unsafe {
     let mut cuda_env = CudaEnv::new(0, 0).unwrap();

     let module = CudaModule::new(b"resources/kernel_test.ptx\0").unwrap();
     let function = module.load_function(b"mul_by_2\0").unwrap();

     let matrix = matrix![[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]];

     let result = apply_function_matrix(&matrix, &cuda_env, function).unwrap();

     assert_eq!(result[0], [2.0, 4.0, 6.0]);
     assert_eq!(result[1], [8.0, 10.0, 12.0]);

     module.free().unwrap();
 }
 ```

 ## Features

 - Initialize a cuda environment
 - Launch common operations on matrices
 - Import and use custom kernel to perform custom operations on matrices
 - Allocate and Free memory in GPU
 - Copy data between Host and Device