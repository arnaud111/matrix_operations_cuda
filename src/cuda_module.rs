use std::collections::HashMap;
use std::error::Error;
use std::fmt::format;
use cuda_driver_sys::{CUfunction, CUmodule, cuModuleGetFunction, cuModuleLoad, CUresult};

pub struct CudaModule {
    hmod: CUmodule,
    functions: HashMap<String, CUfunction>
}

impl CudaModule {

    pub unsafe fn new(path: &[u8]) -> Result<CudaModule, Box<dyn Error>> {
        let mut hmod: CUmodule = std::ptr::null_mut();
        let result = cuModuleLoad(&mut hmod, path.as_ptr() as *const _);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error loading module : {:?}", result).into());
        }
        Ok(CudaModule {
            hmod,
            functions: HashMap::new()
        })
    }

    pub unsafe fn load_function(&mut self, name: &[u8]) -> Result<(), Box<dyn Error>> {
        let mut hfunc: CUfunction = std::ptr::null_mut();
        let result = cuModuleGetFunction(&mut hfunc, self.hmod, name.as_ptr() as *const _);
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("Error loading function: {:?}", result).into());
        }
        self.functions.insert(String::from_utf8(name.to_vec())?, hfunc);
        Ok(())
    }
}