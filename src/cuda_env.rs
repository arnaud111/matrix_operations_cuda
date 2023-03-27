use std::collections::HashMap;
use std::error::Error;
use std::ffi::{c_int, c_uint, c_void};
use std::mem::size_of;
use cuda_driver_sys::{CUcontext, cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxPushCurrent_v2, cuCtxSynchronize, CUdevice, cuDeviceGet, CUdeviceptr, cuInit, cuLaunchKernel, cuMemAlloc_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemFree_v2, cuMemsetD8_v2, CUresult};
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
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn new(flags: c_uint, ordinal: c_int) -> Result<CudaEnv, Box<dyn Error>> {
        let mut ctx: CUcontext = std::ptr::null_mut();
        let mut device: CUdevice = CUdevice::default();
        let modules = HashMap::new();

        let mut result = cuInit(flags);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error initializing CUDA : {:?}", result).into());
        }

        result = cuDeviceGet(&mut device, ordinal);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error getting CUDA device : {:?}", result).into());
        }

        result = cuCtxCreate_v2(&mut ctx, 0, device);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error creating CUDA context : {:?}", result).into());
        }

        result = cuCtxPushCurrent_v2(ctx);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error pushing CUDA context : {:?}", result).into());
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
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let functions = vec!["add".to_string()];
    ///     cuda_env.load_module("resources/kernel_float.ptx".to_string(), "kernel".to_string(), functions).unwrap();
    ///
    ///     cuda_env.free().unwrap();
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

    /// Launch a CUDA kernel
    ///
    /// # Example
    ///
    /// ```
    /// use std::ffi::c_void;
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let functions = vec!["add_scalar".to_string()];
    ///     cuda_env.load_module("resources/kernel_float.ptx".to_string(), "kernel".to_string(), functions).unwrap();
    ///
    ///     let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    ///     let device_ptr_in = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///     cuda_env.copy_host_to_device(device_ptr_in, &data).unwrap();
    ///
    ///     let device_ptr_out = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///     cuda_env.set_empty_device_data(device_ptr_out, data.len() * size_of::<f32>()).unwrap();
    ///
    ///     let scalar = 1.0f32;
    ///     let args = [
    ///         &device_ptr_in as *const _ as *mut c_void,
    ///         &device_ptr_out as *const _ as *mut c_void,
    ///         &scalar as *const _ as *mut c_void
    ///     ];
    ///
    ///     cuda_env.launch("kernel".to_string(), "add_scalar".to_string(), &args, (1, 1, 1), (data.len() as u32, 1, 1)).unwrap();
    ///
    ///     let mut result = [0.0f32; 4];
    ///     cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();
    ///
    ///     assert_eq!(result, [2.0f32, 3.0f32, 4.0f32, 5.0f32]);
    ///
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
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

    /// Allocate memory on device
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    ///     let device_ptr = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///
    ///     cuda_env.free_data(device_ptr).unwrap();
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn allocate(&mut self, size: usize) -> Result<CUdeviceptr, Box<dyn Error>> {
        let mut device_ptr: CUdeviceptr = CUdeviceptr::default();
        let result = cuMemAlloc_v2(&mut device_ptr, size);

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error allocating memory : {:?}", result).into());
        }

        Ok(device_ptr)
    }

    /// Copy data from host to device
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    ///     let device_ptr = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///     cuda_env.copy_host_to_device(device_ptr, &data).unwrap();
    ///
    ///     cuda_env.free_data(device_ptr).unwrap();
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn copy_host_to_device<T>(&self, device_ptr: CUdeviceptr, data: &[T]) -> Result<(), Box<dyn Error>> {
        let result = cuMemcpyHtoD_v2(device_ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>());

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error copy data into device : {:?}", result).into());
        }

        Ok(())
    }

    /// Copy data from device to host
    ///
    /// # Example
    ///
    /// ```
    /// use std::ffi::c_void;
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let functions = vec!["add_scalar".to_string()];
    ///     cuda_env.load_module("resources/kernel_float.ptx".to_string(), "kernel".to_string(), functions).unwrap();
    ///
    ///     let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    ///     let device_ptr_in = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///     cuda_env.copy_host_to_device(device_ptr_in, &data).unwrap();
    ///
    ///     let device_ptr_out = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///     cuda_env.set_empty_device_data(device_ptr_out, data.len() * size_of::<f32>()).unwrap();
    ///
    ///     let scalar = 1.0f32;
    ///     let args = [
    ///         &device_ptr_in as *const _ as *mut c_void,
    ///         &device_ptr_out as *const _ as *mut c_void,
    ///         &scalar as *const _ as *mut c_void
    ///     ];
    ///
    ///     cuda_env.launch("kernel".to_string(), "add_scalar".to_string(), &args, (1, 1, 1), (data.len() as u32, 1, 1)).unwrap();
    ///
    ///     let mut result = [0.0f32; 4];
    ///     cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();
    ///
    ///     assert_eq!(result, [2.0f32, 3.0f32, 4.0f32, 5.0f32]);
    ///
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn copy_device_to_host<T>(&self, data: &mut [T], device_ptr: CUdeviceptr) -> Result<(), Box<dyn Error>> {
        let result = cuMemcpyDtoH_v2(data.as_mut_ptr() as *mut c_void, device_ptr, data.len() * size_of::<T>());

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error copy data into device : {:?}", result).into());
        }

        Ok(())
    }

    /// Set device data to 0
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    ///     let device_ptr = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///     cuda_env.set_empty_device_data(device_ptr, data.len() * size_of::<f32>()).unwrap();
    ///
    ///     cuda_env.free_data(device_ptr).unwrap();
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn set_empty_device_data(&self, device_ptr: CUdeviceptr, size: usize) -> Result<(), Box<dyn Error>> {
        let result = cuMemsetD8_v2(device_ptr, 0, size);

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error setting empty data into device : {:?}", result).into());
        }

        Ok(())
    }

    /// Free CUDA environment
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let data = [1.0f32, 2.0f32, 3.0f32];
    ///     let device_ptr_1 = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///     let device_ptr_2 = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///
    ///     let device_ptrs = vec![device_ptr_1, device_ptr_2];
    ///
    ///     cuda_env.free_all_data(device_ptrs.as_slice()).unwrap();
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn free_all_data(&self, device_ptrs: &[CUdeviceptr]) -> Result<(), Box<dyn Error>> {
        for device_ptr in device_ptrs {
            self.free_data(*device_ptr)?;
        }
        Ok(())
    }

    /// Free CUDA data
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let data = [1.0f32, 2.0f32, 3.0f32];
    ///     let device_ptr = cuda_env.allocate(data.len() * size_of::<f32>()).unwrap();
    ///
    ///     cuda_env.free_data(device_ptr).unwrap();
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn free_data(&self, device_ptr: CUdeviceptr) -> Result<(), Box<dyn Error>> {
        let result = cuMemFree_v2(device_ptr);

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error freeing data : {:?}", result).into());
        }

        Ok(())
    }

    /// Free CUDA environment
    ///
    /// # Example
    ///
    /// ```
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn free(&mut self) -> Result<(), Box<dyn Error>> {
        for (_, module) in self.modules.iter_mut() {
            module.free()?;
        }
        cuCtxDestroy_v2(self.ctx);
        Ok(())
    }

    /// Free CUDA module
    ///
    /// # Example
    ///
    /// ```
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    ///
    /// let mut cuda_env: CudaEnv;
    ///
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let functions = vec!["add".to_string()];
    ///     cuda_env.load_module("resources/kernel_float.ptx".to_string(), "kernel".to_string(), functions).unwrap();
    ///
    ///     cuda_env.free_module("kernel".to_string()).unwrap();
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn free_module(&mut self, name: String) -> Result<(), Box<dyn Error>> {
        let mut module = self.modules.remove(&name).unwrap();
        module.free()?;
        Ok(())
    }
}
