use std::error::Error;
use matrix_operations::Matrix;
use crate::cuda_env::CudaEnv;
use crate::cuda_module::CudaModule;

pub mod cuda_module;
pub mod cuda_env;

pub unsafe fn add_scalar(matrix: &Matrix<f32>, scalar: f32, cuda_env: &CudaEnv) -> Result<Matrix<f32>, Box<dyn Error>> {

    let module = CudaModule::new(b"resources/kernel.ptx\0")?;
    let function = module.load_function(b"add_scalar\0")?;
    let matrix_data = matrix.as_slice();

    let max_threads_per_block = cuda_env.get_max_threads_per_block();
    let max_blocks_per_grid = cuda_env.get_max_block_per_grid();
    let (block_dim, grid_dim) = CudaEnv::get_block_and_grid_dim(matrix_data.len(), max_threads_per_block, max_blocks_per_grid);

    Ok(matrix.clone())
}

