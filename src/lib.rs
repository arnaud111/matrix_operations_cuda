use crate::cuda_env::CudaEnv;

pub mod cuda_module;
pub mod cuda_env;

/// Load the CUDA float kernel for matrix operations
///
/// # Example
///
/// ```
/// use matrix_operations_cuda::cuda_env::CudaEnv;
/// use matrix_operations_cuda::init_float_kernel;
///
/// unsafe {
///     let mut env = CudaEnv::new(0, 0).unwrap();
///     init_float_kernel(&mut env).unwrap();
///     env.free().unwrap();
/// }
/// ```
pub unsafe fn init_float_kernel(env: &mut CudaEnv) -> Result<(), Box<dyn std::error::Error>> {
    let functions = vec![
        "add".to_string(),
        "sub".to_string(),
        "dot".to_string(),
        "add_scalar".to_string(),
        "sub_scalar".to_string(),
        "mul_scalar".to_string(),
        "div_scalar".to_string(),
        "scalar_sub".to_string(),
        "scalar_div".to_string()
    ];
    env.load_module("resources/kernel_float.ptx".to_string(), "kernel_float".to_string(), functions)?;
    Ok(())
}

/// Load the CUDA int kernel for matrix operations
///
/// # Example
///
/// ```
/// use matrix_operations_cuda::cuda_env::CudaEnv;
/// use matrix_operations_cuda::init_int_kernel;
///
/// unsafe {
///     let mut env = CudaEnv::new(0, 0).unwrap();
///     init_int_kernel(&mut env).unwrap();
///     env.free().unwrap();
/// }
/// ```
pub unsafe fn init_int_kernel(env: &mut CudaEnv) -> Result<(), Box<dyn std::error::Error>> {
    let functions = vec![
        "add".to_string(),
        "sub".to_string(),
        "dot".to_string(),
        "add_scalar".to_string(),
        "sub_scalar".to_string(),
        "mul_scalar".to_string(),
        "div_scalar".to_string(),
        "scalar_sub".to_string(),
        "scalar_div".to_string()
    ];
    env.load_module("resources/kernel_int.ptx".to_string(), "kernel_int".to_string(), functions)?;
    Ok(())
}
