use std::error::Error;
use std::ffi::{c_int, c_uint, c_void};
use std::mem::size_of;
use cuda_driver_sys::{CUcontext, cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxPushCurrent_v2, cuCtxSynchronize, CUdevice, CUdevice_attribute_enum, cuDeviceGet, cuDeviceGetAttribute, CUdeviceptr, CUfunction, cuInit, cuLaunchKernel, cuMemAlloc_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemFree_v2, cuMemsetD8_v2, CUresult};

pub struct CudaEnv {
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
            ctx,
            device
        })
    }

    /// Get max threads per block
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
    ///     let max_threads_per_block = cuda_env.get_max_threads_per_block();
    ///     println!("Max threads per block: {}", max_threads_per_block);
    ///
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn get_max_threads_per_block(&self) -> u32 {
        let mut max_threads_per_block: i32 = 0;
        cuDeviceGetAttribute(&mut max_threads_per_block as *mut c_int, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self.device);
        max_threads_per_block as u32
    }

    /// Get max block per grid
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
    ///     let max_block_per_grid = cuda_env.get_max_block_per_grid();
    ///     println!("Max block per grid: {}", max_block_per_grid);
    ///
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn get_max_block_per_grid(&self) -> u32 {
        let mut max_block_per_grid: i32 = 0;
        cuDeviceGetAttribute(&mut max_block_per_grid as *mut c_int, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, self.device);
        max_block_per_grid as u32
    }

    /// Get grid_size and block_size
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
    ///     let max_threads_per_block = cuda_env.get_max_threads_per_block(); // 1024
    ///     let max_block_per_grid = cuda_env.get_max_block_per_grid();
    ///
    ///     let (block_dim, grid_dim) = CudaEnv::get_block_and_grid_dim(2000, max_threads_per_block, max_block_per_grid);
    ///
    ///     assert_eq!(block_dim, (32, 32, 1));
    ///     assert_eq!(grid_dim, (2, 1, 1));
    ///
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn get_block_and_grid_dim(data_len: usize, max_threads: u32, max_block: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
        if is_perfect_square(max_threads) {
            let block_dim = (max_threads as f32).sqrt() as u32;
            let grid_dim = (data_len as f32 / max_threads as f32).ceil() as u32;

            if grid_dim <= max_block {
                return ((block_dim, block_dim, 1), (grid_dim, 1, 1));
            }
        } else {
            let block_dim = max_threads;
            let grid_dim = (data_len as f32 / max_threads as f32).ceil() as u32;

            if grid_dim <= max_block {
                return ((block_dim, 1, 1), (grid_dim, 1, 1));
            }
        }

        ((1, 1, 1), (1, 1, 1))
    }

    /// Launch a CUDA kernel
    ///
    /// # Example
    ///
    /// ```
    /// use std::ffi::c_void;
    /// use std::mem::size_of;
    /// use matrix_operations_cuda::cuda_env::CudaEnv;
    /// use matrix_operations_cuda::cuda_module::CudaModule;
    ///
    /// let mut cuda_env: CudaEnv;
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
    ///     let function = module.load_function(b"add_scalar\0").unwrap();
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
    ///     cuda_env.launch(function, &args, (1, 1, 1), (data.len() as u32, 1, 1)).unwrap();
    ///
    ///     let mut result = [0.0f32; 4];
    ///     cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();
    ///
    ///     assert_eq!(result, [2.0f32, 3.0f32, 4.0f32, 5.0f32]);
    ///
    ///     cuda_env.free_data(device_ptr_in).unwrap();
    ///     cuda_env.free_data(device_ptr_out).unwrap();
    ///     module.free().unwrap();
    ///     cuda_env.free().unwrap();
    /// }
    /// ```
    pub unsafe fn launch(&self, function: CUfunction, args: &[*mut c_void], grid_size: (u32, u32, u32), bloc_size: (u32, u32, u32)) -> Result<(), Box<dyn Error>> {

        let result = cuLaunchKernel(
            function,
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
    /// use matrix_operations_cuda::cuda_module::CudaModule;
    ///
    /// let mut cuda_env: CudaEnv;
    /// unsafe {
    ///     cuda_env = CudaEnv::new(0, 0).unwrap();
    ///
    ///     let module = CudaModule::new(b"resources/kernel.ptx\0").unwrap();
    ///     let function = module.load_function(b"add_scalar\0").unwrap();
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
    ///     cuda_env.launch(function, &args, (1, 1, 1), (data.len() as u32, 1, 1)).unwrap();
    ///
    ///     let mut result = [0.0f32; 4];
    ///     cuda_env.copy_device_to_host(&mut result, device_ptr_out).unwrap();
    ///
    ///     assert_eq!(result, [2.0f32, 3.0f32, 4.0f32, 5.0f32]);
    ///
    ///     cuda_env.free_data(device_ptr_in).unwrap();
    ///     cuda_env.free_data(device_ptr_out).unwrap();
    ///     module.free().unwrap();
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
        cuCtxDestroy_v2(self.ctx);
        Ok(())
    }
}

fn is_perfect_square(num: u32) -> bool {
    let sqrt_num = (num as f64).sqrt() as u32;
    return sqrt_num * sqrt_num == num;
}
