//! Module for loading cuda modules
//!
//! # Usage
//!
//! ```
//! use cuda_driver_sys::*;
//! use matrix_operations_cuda::cuda_env::CudaEnv;
//! use matrix_operations_cuda::cuda_module::CudaModule;
//!
//! unsafe {
//!     let mut cuda_env = CudaEnv::new(0, 0).unwrap();
//!     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
//!     let function = module.load_function(b"add\0").unwrap();
//!     module.free().unwrap();
//!     cuda_env.free().unwrap();
//! }
//! ```
use std::error::Error;
use cuda_driver_sys::{CUfunction, CUmodule, cuModuleGetFunction, cuModuleLoad, cuModuleUnload, CUresult};

/// Cuda module struct
pub struct CudaModule {
    module: CUmodule,
}

impl CudaModule {

    /// Create a new CudaModule from a ptx file
    ///
    /// # Example
    ///
    /// ```
    /// use cuda_driver_sys::*;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    /// use matrix_operations_cuda::cuda_module::CudaModule;
    ///
    /// unsafe {
    ///     let cuda_env = CudaEnv::new(0, 0);
    ///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
    ///     module.free().unwrap();
    /// }
    /// ```
    ///
    pub unsafe fn new(path: &[u8]) -> Result<CudaModule, Box<dyn Error>> {
        let mut module: CUmodule = std::ptr::null_mut();
        let result = cuModuleLoad(&mut module, path.as_ptr() as *const _);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error loading module : {:?}", result).into());
        }
        Ok(CudaModule {
            module,
        })
    }

    /// Load a function from the module
    ///
    /// # Example
    ///
    /// ```
    /// use cuda_driver_sys::*;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    /// use matrix_operations_cuda::cuda_module::CudaModule;
    ///
    /// unsafe {
    ///     let cuda_env = CudaEnv::new(0, 0);
    ///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
    ///     let function = module.load_function(b"add\0").unwrap();
    ///     module.free().unwrap();
    /// }
    /// ```
    pub unsafe fn load_function(&self, name: &[u8]) -> Result<CUfunction, Box<dyn Error>> {
        let mut function: CUfunction = std::ptr::null_mut();
        let result = cuModuleGetFunction(&mut function, self.module, name.as_ptr() as *const _);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error loading function: {:?}", result).into());
        }
        Ok(function)
    }

    /// Free the module
    ///
    /// # Example
    ///
    /// ```
    /// use cuda_driver_sys::*;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    /// use matrix_operations_cuda::cuda_module::CudaModule;
    ///
    /// unsafe {
    ///     let cuda_env = CudaEnv::new(0, 0);
    ///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
    ///     module.free().unwrap();
    /// }
    /// ```
    pub unsafe fn free(&self) -> Result<(), Box<dyn Error>> {
        let result = cuModuleUnload(self.module);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error unloading module : {:?}", result).into());
        }
        Ok(())
    }
}