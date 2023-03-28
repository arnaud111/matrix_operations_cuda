use std::error::Error;
use std::ffi::c_void;
use std::mem::size_of;
use matrix_operations::Matrix;
use crate::cuda_env::CudaEnv;
use crate::cuda_module::CudaModule;

pub mod cuda_module;
pub mod cuda_env;

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
///     let result = add_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [3.00f32, 4.00f32]);
///     assert_eq!(result[1], [5.00f32, 6.00f32]);
/// }
pub unsafe fn add_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"add_scalar\0")?;
    let mut matrix_data = matrix.as_slice();

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

    Matrix::from_slice(result.as_slice(), matrix.shape())
}

