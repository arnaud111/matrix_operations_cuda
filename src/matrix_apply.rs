use std::error::Error;
use std::ffi::c_void;
use std::mem::size_of;
use cuda_driver_sys::CUfunction;
use matrix_operations::Matrix;
use crate::CudaEnv;

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

    Matrix::from_slice(result.as_slice(), matrix.shape())
}

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

    Matrix::from_slice(result.as_slice(), matrix1.shape())
}

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

    Matrix::from_slice(result.as_slice(), returned_shape)
}

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

    Matrix::from_slice(result.as_slice(), matrix.shape())
}

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

    Matrix::from_slice(result.as_slice(), returned_shape)
}