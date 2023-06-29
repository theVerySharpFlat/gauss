use std::{ffi::CString, ptr, str::FromStr, sync::Arc};

use ash::vk::{
    self, ComputePipelineCreateInfo, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
    DescriptorPoolSize, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags,
    DescriptorSetLayoutCreateInfo, DescriptorType, PipelineCache, PipelineCreateFlags,
    PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, PipelineShaderStageCreateFlags,
    PipelineShaderStageCreateInfo, ShaderModule, ShaderModuleCreateFlags, ShaderModuleCreateInfo,
    ShaderStageFlags, StructureType,
};

use shaderc;

use super::ComputeManager;

#[derive(Clone, Copy, Debug)]
pub enum PipelineCreateError {
    InvalidShader,
    DescriptorSetLayoutCreationFailure,
    PipelineLayoutCreationFailure,
    PipelineCreationFailure,
    DescriptorPoolCreationFailure,
    DescriptorSetAllocationFailure,
}

pub struct Pipeline {
    pub(super) pipeline: vk::Pipeline,
    pub(super) pipeline_layout: vk::PipelineLayout,

    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) descriptor_pool: vk::DescriptorPool,

    parent: Arc<ComputeManager>,
}

pub struct Program {
    shader_module: ShaderModule,
    shader_name: String,
}

#[derive(Debug, Clone)]
pub enum ProgramCompilationError {
    SPIRVCompilationError(String),
    ModuleCreationError(String),
}

impl ComputeManager {
    pub fn compile_program(
        self: &Self,
        shader: &str,
        name: &str,
        optimize: bool,
    ) -> Result<Program, ProgramCompilationError> {
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        if !optimize {
            options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        }

        let result = match compiler.compile_into_spirv(
            shader,
            shaderc::ShaderKind::Compute,
            name,
            "main",
            Some(&options),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Err(ProgramCompilationError::SPIRVCompilationError(format!(
                    "Shader compilation of \"{}\" failed with error \"{}\"",
                    name, e
                )));
            }
        };

        let shader_module_create_info = ShaderModuleCreateInfo {
            s_type: StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ShaderModuleCreateFlags::empty(),
            code_size: result.as_binary().len() * 4,
            p_code: result.as_binary().as_ptr(),
        };

        let shader_module = unsafe {
            match self
                .device_info
                .device
                .create_shader_module(&shader_module_create_info, None)
            {
                Ok(r) => r,
                Err(e) => return Err(ProgramCompilationError::ModuleCreationError(e.to_string())),
            }
        };

        Ok(Program {
            shader_module,
            shader_name: String::from_str(name).unwrap(),
        })
    }

    pub fn build_pipeline(
        self: Arc<Self>,
        program: Program,
        n_tensors: u32,
    ) -> Result<Pipeline, PipelineCreateError> {
        let mut descriptor_set_bindings: Vec<DescriptorSetLayoutBinding> = Vec::new();
        for i in 0..n_tensors {
            descriptor_set_bindings.push(DescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
            });
        }

        let create_info = DescriptorSetLayoutCreateInfo {
            s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: DescriptorSetLayoutCreateFlags::empty(),
            binding_count: descriptor_set_bindings.len() as u32,
            p_bindings: descriptor_set_bindings.as_ptr(),
        };

        let descriptor_set_layout = unsafe {
            match self
                .device_info
                .device
                .create_descriptor_set_layout(&create_info, None)
            {
                Ok(l) => l,
                Err(e) => {
                    log::error!("Failed to create descriptor set layout! Error: {}", e);
                    return Err(PipelineCreateError::DescriptorSetLayoutCreationFailure);
                }
            }
        };

        let pipeline_layout_create_info = PipelineLayoutCreateInfo {
            s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: PipelineLayoutCreateFlags::empty(),
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
        };

        let pipeline_layout = unsafe {
            match self
                .device_info
                .device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
            {
                Ok(l) => l,
                Err(e) => {
                    log::error!("Failed to create pipeline layout! Error: {}", e);
                    return Err(PipelineCreateError::PipelineLayoutCreationFailure);
                }
            }
        };

        let name_cstring = CString::new("main").unwrap();
        let shader_stage_create_info = PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: PipelineShaderStageCreateFlags::empty(),
            stage: ShaderStageFlags::COMPUTE,
            module: program.shader_module,
            p_name: name_cstring.as_ptr(),
            p_specialization_info: ptr::null(),
        };

        let pipeline_create_info = ComputePipelineCreateInfo {
            s_type: StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: PipelineCreateFlags::empty(),
            stage: shader_stage_create_info,
            layout: pipeline_layout.clone(),
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
        };

        let pipeline = unsafe {
            match self.device_info.device.create_compute_pipelines(
                PipelineCache::null(),
                &[pipeline_create_info],
                None,
            ) {
                Ok(p) => p[0],
                Err((_, e)) => {
                    log::error!("Failed to create pipeline! Error {}", e);
                    return Err(PipelineCreateError::PipelineCreationFailure);
                }
            }
        };

        unsafe {
            self.device_info
                .device
                .destroy_shader_module(program.shader_module, None)
        }

        let pool_size = DescriptorPoolSize {
            ty: DescriptorType::STORAGE_BUFFER,
            descriptor_count: n_tensors as u32,
        };

        let descriptor_pool_create_info = DescriptorPoolCreateInfo {
            s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: 10,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };

        let descriptor_pool = unsafe {
            match self
                .device_info
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
            {
                Ok(p) => p,
                Err(e) => {
                    log::error!("Failed to create descriptor pool! Error: {}", e);
                    return Err(PipelineCreateError::DescriptorPoolCreationFailure);
                }
            }
        };

        Ok(Pipeline {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            parent: self.clone(),
        })
    }
}

impl<'a> Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.parent
                .device_info
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.parent
                .device_info
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.parent
                .device_info
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.parent
                .device_info
                .device
                .destroy_pipeline(self.pipeline, None);
        }
    }
}
