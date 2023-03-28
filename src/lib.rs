use std::error::Error;
use std::ffi::c_void;
use std::mem::size_of;
use cuda_driver_sys::CUfunction;
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
///     let result = add_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 + 2.0);
///     }
/// }
/// ```
pub unsafe fn add_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"add_scalar\0")?;
    function_scalar(matrix, scalar, cuda_env, function)
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
///     let result = sub_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [-1.00f32, 0.00f32]);
///     assert_eq!(result[1], [1.00f32, 2.00f32]);
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
///     let result = sub_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 - 2.0);
///     }
/// }
/// ```
pub unsafe fn sub_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"sub_scalar\0")?;
    function_scalar(matrix, scalar, cuda_env, function)
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
///     let result = mul_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [2.00f32, 4.00f32]);
///     assert_eq!(result[1], [6.00f32, 8.00f32]);
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
///     let result = mul_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 * 2.0);
///     }
/// }
/// ```
pub unsafe fn mul_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"mul_scalar\0")?;
    function_scalar(matrix, scalar, cuda_env, function)
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
///     let result = div_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [0.50f32, 1.00f32]);
///     assert_eq!(result[1], [1.50f32, 2.00f32]);
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
///     let result = div_scalar(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], i as f32 / 2.0);
///     }
/// }
/// ```
pub unsafe fn div_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"div_scalar\0")?;
    function_scalar(matrix, scalar, cuda_env, function)
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
///     let result = scalar_sub(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [1.00f32, 0.00f32]);
///     assert_eq!(result[1], [-1.00f32, -2.00f32]);
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
///     let result = scalar_sub(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 - i as f32);
///     }
/// }
/// ```
pub unsafe fn scalar_sub(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"scalar_sub\0")?;
    function_scalar(matrix, scalar, cuda_env, function)
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
///     let result = scalar_div(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [2.00f32, 1.00f32]);
///     assert_eq!(result[1], [0.6666667f32, 0.5f32]);
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
///     let result = scalar_div(&matrix, 2.0, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 / i as f32);
///     }
/// }
/// ```
pub unsafe fn scalar_div(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"scalar_div\0")?;
    function_scalar(matrix, scalar, cuda_env, function)
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
///     let result = add_matrices(&matrix1, &matrix2, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [11.00f32, 13.00f32]);
///     assert_eq!(result[1], [15.00f32, 17.00f32]);
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
///     let result = add_matrices(&matrix1, &matrix1, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 2.0 * i as f32);
///     }
/// }
/// ```
pub unsafe fn add_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"add\0")?;
    function_two_matrices(matrix1, matrix2, cuda_env, function)
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
///     let result = sub_matrices(&matrix1, &matrix2, &mut cuda_env).unwrap();
///
///     assert_eq!(result[0], [-9.0f32, -9.00f32]);
///     assert_eq!(result[1], [-9.00f32, -9.00f32]);
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
///     let result = sub_matrices(&matrix1, &matrix1, &mut cuda_env).unwrap();
///
///     let data_result = result.as_slice();
///     for i in 0..data.len() {
///         assert_eq!(data_result[i], 0.0f32);
///     }
/// }
/// ```
pub unsafe fn sub_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &mut CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"sub\0")?;
    function_two_matrices(matrix1, matrix2, cuda_env, function)
}

unsafe fn function_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &mut CudaEnv, function: CUfunction) -> Result<Matrix<f32>, Box<dyn Error>> {
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

unsafe fn function_two_matrices(matrix1: &Matrix<f32>, matrix2: &Matrix<f32>, cuda_env: &mut CudaEnv, function: CUfunction) -> Result<Matrix<f32>, Box<dyn Error>> {
    let mut matrix1_data = matrix1.as_slice();
    let mut matrix2_data = matrix2.as_slice();

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
