use std::collections::HashMap;
use std::error::Error;
use std::ffi::{c_int, c_uint};
use cuda_driver_sys::{CUcontext, cuCtxCreate_v2, cuCtxPushCurrent_v2, CUdevice, cuDeviceGet, cuInit, CUresult};
use crate::cuda_module::CudaModule;

pub struct CudaEnv {
    modules: HashMap<String, CudaModule>,
    ctx: CUcontext,
    device: CUdevice
}

impl CudaEnv {

    /// Initialize CUDA environment
    ///
    /// # Example
    ///
    /// ```
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    /// let cuda_env: CudaEnv;
    /// unsafe {
    ///   cuda_env = CudaEnv::new(0, 0).unwrap();
    /// }
    /// ```
    pub unsafe fn new(flags: c_uint, ordinal: c_int) -> Result<CudaEnv, Box<dyn Error>> {
        let mut ctx: CUcontext = std::ptr::null_mut();
        let mut device: CUdevice = CUdevice::default();
        let modules = HashMap::new();

        if cuInit(flags) != CUresult::CUDA_SUCCESS {
            return Err("Error initializing CUDA".into());
        }
        if cuDeviceGet(&mut device, ordinal) != CUresult::CUDA_SUCCESS {
            return Err("Error getting CUDA device".into());
        }
        if cuCtxCreate_v2(&mut ctx, 0, device) != CUresult::CUDA_SUCCESS {
            return Err("Error creating CUDA context".into());
        }
        if cuCtxPushCurrent_v2(ctx) != CUresult::CUDA_SUCCESS {
            return Err("Error pushing CUDA context".into());
        }

        Ok(CudaEnv {
            modules,
            ctx,
            device
        })
    }

    /// Load a CUDA module
    ///
    /// # Example
    ///
    /// ```
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    /// unsafe {
    ///   cuda_env = CudaEnv::new(0, 0).unwrap();
    ///   let functions = vec!["add".to_string()];
    ///   cuda_env.load_module("resources/kernel.ptx".to_string(), "kernel".to_string(), functions).unwrap();
    /// }
    /// ```
    pub unsafe fn load_module(&mut self, path: String, name: String, functions: Vec<String>) -> Result<(), Box<dyn Error>> {
        let mut module = CudaModule::new(path.as_bytes())?;
        for function in functions {
            module.load_function(function.as_bytes())?;
        }
        self.modules.insert(name, module);
        Ok(())
    }
}
