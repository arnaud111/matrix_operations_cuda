use std::collections::HashMap;
use std::error::Error;
use std::ffi::{c_int, c_uint, c_void};
use std::mem::size_of;
use cuda_driver_sys::{CUcontext, cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxPushCurrent_v2, cuCtxSynchronize, CUdevice, cuDeviceGet, CUdeviceptr, cuInit, cuLaunchKernel, cuMemAlloc_v2, cuMemcpyHtoD_v2, cuMemFree_v2, cuMemsetD8_v2, CUresult};
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

    pub unsafe fn launch(&self, module: String, function: String, args: &[*mut c_void], grid_size: (c_uint, c_uint, c_uint), bloc_size: (c_uint, c_uint, c_uint)) -> Result<(), Box<dyn Error>> {

        let result = cuLaunchKernel(
            self.modules[&module].functions[&function],
            grid_size.0, grid_size.1, grid_size.2,
            bloc_size.0, bloc_size.1, bloc_size.2,
            0,
            std::ptr::null_mut(),
            args.as_ptr() as *mut _,
            std::ptr::null_mut()
        );

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error launching kernel : {:?}", result).into());
        }

        let result = cuCtxSynchronize();

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error synchronizing kernel : {:?}", result).into());
        }

        Ok(())
    }

    pub unsafe fn allocate<T>(&mut self, data: &[T]) -> Result<CUdeviceptr, Box<dyn Error>> {
        let mut device_ptr: CUdeviceptr = CUdeviceptr::default();
        let result = cuMemAlloc_v2(&mut device_ptr, data.len() * size_of::<T>());

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error allocating memory : {:?}", result).into());
        }

        Ok(device_ptr)
    }

    pub unsafe fn copy_host_to_device<T>(&self, device_ptr: CUdeviceptr, data: &[T]) -> Result<(), Box<dyn Error>> {
        let result = cuMemcpyHtoD_v2(device_ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>());

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error copy data into device : {:?}", result).into());
        }

        Ok(())
    }

    pub unsafe fn free_all_data(&self, device_ptrs: &[CUdeviceptr]) -> Result<(), Box<dyn Error>> {
        for device_ptr in device_ptrs {
            self.free_data(*device_ptr)?;
        }
        Ok(())
    }

    pub unsafe fn free_data(&self, device_ptr: CUdeviceptr) -> Result<(), Box<dyn Error>> {
        let result = cuMemFree_v2(CUdeviceptr);

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error freeing data : {:?}", result).into());
        }

        Ok(())
    }

    pub unsafe fn free(&mut self) -> Result<(), Box<dyn Error>> {
        for (_, module) in self.modules.iter_mut() {
            module.free()?;
        }
        cuCtxDestroy_v2(self.ctx);
        Ok(())
    }

    pub unsafe fn free_module(&mut self, name: String) -> Result<(), Box<dyn Error>> {
        let mut module = self.modules.remove(&name).unwrap();
        module.free()?;
        Ok(())
    }
}
