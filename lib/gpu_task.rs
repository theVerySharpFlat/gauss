use std::{
    collections::HashMap,
    ffi::c_void,
    ptr,
    sync::{Arc, RwLock},
};

use ash::vk::{
    AccessFlags, BufferCopy, BufferUsageFlags, CommandBuffer, DependencyFlags,
    DescriptorBufferInfo, DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
    DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo, DescriptorType, Fence,
    MemoryBarrier, PipelineBindPoint, PipelineStageFlags, StructureType, WriteDescriptorSet, DescriptorPoolResetFlags,
};

use super::{
    allocation_strategy::Allocator, allocation_strategy::Buffer, command_buffer_util,
    device::DeviceInfo, pipeline::Pipeline, ComputeManager, Tensor,
};

struct TensorBufferBacking {
    pub(super) gpu_buffer: Buffer,
    pub(super) staging_buffer: Buffer,

    pub(super) readback_buffer: Option<Buffer>,
}

pub struct GPUTask {
    command_buffer: CommandBuffer,
    device_info: DeviceInfo,
    buffers: HashMap<u32, TensorBufferBacking>,
    descriptor_set: DescriptorSet,
    parent_descriptor_pool: DescriptorPool,
    allocator: Arc<RwLock<Allocator>>,

    _parent: Arc<ComputeManager>,
}

pub struct GPUTaskInProcess {
    errno: Option<GPUTaskRecordingError>,
    task: Option<GPUTask>,
}

#[derive(Debug, Clone, Copy)]
pub struct WorkGroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct GPUSyncPrimitive<'a> {
    pub(super) fence: Fence,

    parent: &'a GPUTask,
}

#[derive(Debug, Clone, Copy)]
pub enum GPUTaskRecordingError {
    CommandBufferAllocationFailure,
    CommandBufferRecordingStartFailure,
    BufferAllocationFailure,
    DescriptorSetAllocationFailure,
    UnknownError,
}

impl ComputeManager {
    pub fn new_task(
        self: Arc<Self>,
        pipeline: &Pipeline,
        bindings: Vec<&Tensor>,
    ) -> GPUTaskInProcess {
        let mut buffer_backing = HashMap::<u32, TensorBufferBacking>::with_capacity(bindings.len());

        // Allocate buffers
        for (_i, binding) in bindings.iter().enumerate() {
            let mut allocator_actual = match self.allocator.write() {
                Ok(a) => a,
                Err(e) => {
                    log::error!("Failed to acquire allocator! Error: {e}");
                    return GPUTaskInProcess {
                        errno: Some(GPUTaskRecordingError::BufferAllocationFailure),
                        task: None,
                    };
                }
            };

            let gpu_buffer = match allocator_actual.allocate_buffer(
                &self.device_info,
                (binding.data().len() * 4) as u64,
                BufferUsageFlags::STORAGE_BUFFER
                    | BufferUsageFlags::TRANSFER_SRC
                    | BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuOnly,
                format!("gpu_only_alloc{{id={}}}", binding.id).as_str(),
                self.device_info.queue_indices.compute_queue.unwrap(),
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("Failed to allocate buffer! Error: {:?}", e);
                    return GPUTaskInProcess {
                        errno: Some(GPUTaskRecordingError::BufferAllocationFailure),
                        task: None,
                    };
                }
            };

            let staging_buffer = match allocator_actual.allocate_buffer(
                &self.device_info,
                (binding.data().len() * 4) as u64,
                BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
                format!("gpu_staging_only_alloc{{id={}}}", binding.id).as_str(),
                self.device_info.queue_indices.compute_queue.unwrap(),
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("Failed to allocate buffer! Error: {:?}", e);
                    return GPUTaskInProcess {
                        errno: Some(GPUTaskRecordingError::BufferAllocationFailure),
                        task: None,
                    };
                }
            };

            let readback_buffer = if binding.readback_enabled {
                Some(
                    match allocator_actual.allocate_buffer(
                        &self.device_info,
                        (binding.data().len() * 4) as u64,
                        BufferUsageFlags::TRANSFER_DST,
                        gpu_allocator::MemoryLocation::CpuToGpu,
                        format!("gpu_staging_only_alloc{{id={}}}", binding.id).as_str(),
                        self.device_info.queue_indices.compute_queue.unwrap(),
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            log::error!("Failed to allocate buffer! Error: {:?}", e);
                            return GPUTaskInProcess {
                                errno: Some(GPUTaskRecordingError::BufferAllocationFailure),
                                task: None,
                            };
                        }
                    },
                )
            } else {
                None
            };

            let backing = TensorBufferBacking {
                gpu_buffer,
                staging_buffer,
                readback_buffer,
            };

            buffer_backing.insert(binding.id, backing);
        }

        let pool_size = DescriptorPoolSize {
            ty: DescriptorType::STORAGE_BUFFER,
            descriptor_count: bindings.len() as u32,
        };

        let descriptor_pool_create_info = DescriptorPoolCreateInfo {
            s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: DescriptorPoolCreateFlags::empty(),
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
                    return GPUTaskInProcess {
                        errno: Some(GPUTaskRecordingError::DescriptorSetAllocationFailure),
                        task: None,
                    };
                }
            }
        };

        let descriptor_set_alloc_info = DescriptorSetAllocateInfo {
            s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &pipeline.descriptor_set_layout,
        };

        let descriptor_set = unsafe {
            match self
                .device_info
                .device
                .allocate_descriptor_sets(&descriptor_set_alloc_info)
            {
                Ok(s) => s,
                Err(e) => {
                    log::error!("Failed to allocate descriptor set! Error: {}", e);
                    return GPUTaskInProcess {
                        errno: Some(GPUTaskRecordingError::DescriptorSetAllocationFailure),
                        task: None,
                    };
                }
            }
        };

        {
            let mut descriptor_writes = Vec::<WriteDescriptorSet>::with_capacity(bindings.len());
            let mut descriptor_write_buffer_infos =
                Vec::<DescriptorBufferInfo>::with_capacity(bindings.len());

            bindings.iter().enumerate().for_each(|(i, binding)| {
                descriptor_write_buffer_infos.push(DescriptorBufferInfo {
                    buffer: buffer_backing
                        .get(&binding.id)
                        .unwrap()
                        .gpu_buffer
                        .buffer,
                    offset: 0,
                    range: (binding.data().len() * 4) as u64,
                });
                descriptor_writes.push(WriteDescriptorSet {
                    s_type: StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set[0],
                    dst_binding: i as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: DescriptorType::STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: &descriptor_write_buffer_infos[i],
                    p_texel_buffer_view: ptr::null(),
                });
            });

            unsafe {
                self.device_info
                    .device
                    .update_descriptor_sets(descriptor_writes.as_slice(), &[]);
            }
        }

        let command_buffer = match command_buffer_util::allocate_command_buffer(
            &self.device_info.device,
            self.device_info.compute_pool,
        ) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Failed to allocate command buffer! Error: {}", e);
                return GPUTaskInProcess {
                    errno: Some(GPUTaskRecordingError::CommandBufferAllocationFailure),
                    task: None,
                };
            }
        };

        match command_buffer_util::begin_command_buffer_recording(
            &self.device_info.device,
            command_buffer,
            false,
        ) {
            Ok(_) => (),
            Err(e) => {
                log::error!("Failed to begin command buffer recording! Error: {}", e);
                return GPUTaskInProcess {
                    errno: Some(GPUTaskRecordingError::CommandBufferRecordingStartFailure),
                    task: None,
                };
            }
        }

        unsafe {
            self.device_info.device.cmd_bind_pipeline(
                command_buffer,
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );

            self.device_info.device.cmd_bind_descriptor_sets(
                command_buffer,
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &[descriptor_set[0]],
                &[],
            );
        }

        GPUTaskInProcess {
            task: Some(GPUTask {
                command_buffer,
                device_info: self.device_info.clone(),
                buffers: buffer_backing,
                descriptor_set: descriptor_set[0],
                parent_descriptor_pool: descriptor_pool,
                allocator: self.allocator.clone(),
                _parent: self.clone(),
            }),
            errno: None,
        }
    }

    pub fn exec_task<'a>(&self, task: &'a GPUTask) -> Option<GPUSyncPrimitive<'a>> {
        let fence = match command_buffer_util::end_and_submit_command_buffer(
            &self.device_info.device,
            task.command_buffer,
            self.device_info.compute_queue,
        ) {
            Ok(f) => f,
            Err(e) => {
                log::error!("Failed to submit command buffer! Error: {}", e);
                return None;
            }
        };

        Some(GPUSyncPrimitive {
            fence,
            parent: task,
        })
    }

    pub fn await_task(&self, sync: &GPUSyncPrimitive, sync_tensors: Vec<&mut Tensor>) {
        unsafe {
            let _ = self
                .device_info
                .device
                .wait_for_fences(&[sync.fence], true, u64::MAX);

            self.device_info.device.destroy_fence(sync.fence, None);
        }

        sync_tensors.into_iter().for_each(|tensor| unsafe {
            let backing = match sync.parent.buffers.get(&tensor.id) {
                Some(b) => b,
                None => {
                    log::error!(
                        "Failed to find backing buffer for tensor! This is an internal issue!"
                    );
                    return;
                }
            };

            let mapped_ptr = backing
                .readback_buffer
                .as_ref()
                .unwrap()
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr() as *mut f32;

            tensor
                .data_mut()
                .as_mut_ptr()
                .copy_from(mapped_ptr as *const f32, tensor.data().len());
        });
    }
}

impl GPUTaskInProcess {
    pub fn op_local_sync_device(self, tensors: Vec<&Tensor>) -> Self {
        if self.task.is_none() || self.errno.is_some() {
            return self;
        }

        tensors.iter().for_each(|tensor| unsafe {
            let backing = match self.task.as_ref().unwrap().buffers.get(&tensor.id) {
                Some(b) => b,
                None => {
                    log::error!(
                        "Failed to find backing buffer for tensor! This is an internal issue!"
                    );
                    return;
                }
            };

            backing
                .staging_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr()
                .copy_from(
                    tensor.data().as_ptr() as *const c_void,
                    tensor.data().len() * 4_usize,
                );

            self.task
                .as_ref()
                .unwrap()
                .device_info
                .device
                .cmd_copy_buffer(
                    self.task.as_ref().unwrap().command_buffer,
                    backing.staging_buffer.buffer,
                    backing.gpu_buffer.buffer,
                    &[BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: (tensor.data().len() * 4) as u64,
                    }],
                );
        });

        unsafe {
            self.task
                .as_ref()
                .unwrap()
                .device_info
                .device
                .cmd_pipeline_barrier(
                    self.task.as_ref().unwrap().command_buffer,
                    PipelineStageFlags::TRANSFER,
                    PipelineStageFlags::COMPUTE_SHADER,
                    DependencyFlags::empty(),
                    &[MemoryBarrier {
                        s_type: StructureType::MEMORY_BARRIER,
                        p_next: ptr::null(),
                        src_access_mask: AccessFlags::MEMORY_WRITE,
                        dst_access_mask: AccessFlags::MEMORY_WRITE | AccessFlags::MEMORY_READ,
                    }],
                    &[],
                    &[],
                );
        }

        self
    }

    pub fn op_pipeline_dispatch(self, work_group: WorkGroupSize) -> Self {
        if self.task.is_none() || self.errno.is_some() {
            return self;
        }

        unsafe {
            self.task.as_ref().unwrap().device_info.device.cmd_dispatch(
                self.task.as_ref().unwrap().command_buffer,
                work_group.x,
                work_group.y,
                work_group.z,
            );
        }

        self
    }

    pub fn op_device_sync_local(self, tensors: Vec<&Tensor>) -> Self {
        if self.task.is_none() || self.errno.is_some() {
            return self;
        }

        unsafe {
            self.task
                .as_ref()
                .unwrap()
                .device_info
                .device
                .cmd_pipeline_barrier(
                    self.task.as_ref().unwrap().command_buffer,
                    PipelineStageFlags::COMPUTE_SHADER,
                    PipelineStageFlags::TRANSFER,
                    DependencyFlags::empty(),
                    &[MemoryBarrier {
                        s_type: StructureType::MEMORY_BARRIER,
                        p_next: ptr::null(),
                        src_access_mask: AccessFlags::MEMORY_WRITE,
                        dst_access_mask: AccessFlags::MEMORY_READ,
                    }],
                    &[],
                    &[],
                )
        }

        tensors.iter().for_each(|tensor| unsafe {
            let backing = match self.task.as_ref().unwrap().buffers.get(&tensor.id) {
                Some(b) => b,
                None => {
                    log::error!(
                        "Failed to find backing buffer for tensor! This is an internal issue!"
                    );
                    return;
                }
            };

            if backing.readback_buffer.is_none() {
                log::error!("Tensor has no readback buffer! Did you enable readback on creation?");
                return;
            }

            self.task
                .as_ref()
                .unwrap()
                .device_info
                .device
                .cmd_copy_buffer(
                    self.task.as_ref().unwrap().command_buffer,
                    backing.gpu_buffer.buffer,
                    backing.readback_buffer.as_ref().unwrap().buffer,
                    &[BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: (tensor.data().len() * 4) as u64,
                    }],
                )
        });

        self
    }

    pub fn finalize(self) -> Result<GPUTask, GPUTaskRecordingError> {
        if self.errno.is_some() {
            Err(self.errno.unwrap())
        } else if self.task.is_some() {
            return Ok(self.task.unwrap());
        } else {
            log::error!("This is an GPU task recording API error! Either you have done something really wrong or the API has a mistake in it that we haven't caught!");
            return Err(GPUTaskRecordingError::UnknownError);
        }
    }
}

impl Drop for GPUTask {
    fn drop(&mut self) {
        unsafe {
            self.device_info.device.free_command_buffers(
                self.device_info.compute_pool,
                &[self.command_buffer],
            );

            let _ = self.device_info.device.reset_descriptor_pool(self.parent_descriptor_pool, DescriptorPoolResetFlags::empty());
            self.device_info.device.destroy_descriptor_pool(self.parent_descriptor_pool, None);

            // Free backing buffers
            self.buffers.iter_mut().for_each(|(_, buffer)| {
                let gpu_alloc = std::mem::take(&mut buffer.gpu_buffer.allocation);
                if let Ok(mut allocator_actual) = self.allocator.write() {
                    let _ = allocator_actual.vulkan_allocator.free(gpu_alloc);
                    self.device_info
                        .device
                        .destroy_buffer(buffer.gpu_buffer.buffer, None);

                    let stage_alloc = std::mem::take(&mut buffer.staging_buffer.allocation);
                    let _ = allocator_actual.vulkan_allocator.free(stage_alloc);
                    self.device_info
                        .device
                        .destroy_buffer(buffer.staging_buffer.buffer, None);

                    if buffer.readback_buffer.is_some() {
                        let readback_alloc = std::mem::take(
                            &mut buffer.readback_buffer.as_mut().unwrap().allocation,
                        );
                        let _ = allocator_actual.vulkan_allocator.free(readback_alloc);
                        self.device_info
                            .device
                            .destroy_buffer(buffer.readback_buffer.as_mut().unwrap().buffer, None);
                    }
                } else {
                    log::error!("Failed to acquire allocator for GPU task!");
                }
            });
        }
    }
}
