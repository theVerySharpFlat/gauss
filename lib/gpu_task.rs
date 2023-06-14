use std::{collections::HashMap, ffi::c_void, ptr};

use ash::vk::{
    AccessFlags, BufferCopy, BufferUsageFlags, CommandBuffer, DependencyFlags,
    DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorType,
    Fence, MemoryBarrier, PipelineBindPoint, PipelineStageFlags, StructureType, WriteDescriptorSet,
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
    allocator: *mut Allocator, // :grimace:
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
}

impl ComputeManager {
    pub fn new_task<'a>(
        &'a mut self,
        pipeline: &'a Pipeline,
        bindings: Vec<&Tensor>,
    ) -> Result<GPUTask, GPUTaskRecordingError> {
        let mut buffer_backing = HashMap::<u32, TensorBufferBacking>::with_capacity(bindings.len());

        // Allocate buffers
        for (_i, binding) in bindings.iter().enumerate() {
            let gpu_buffer = match self.allocator.allocate_buffer(
                &self.device_info,
                (binding.data().len() * 4) as u64,
                BufferUsageFlags::STORAGE_BUFFER
                    | BufferUsageFlags::TRANSFER_SRC
                    | BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuOnly,
                "gpu_only_alloc",
                self.device_info.queue_indices.compute_queue.unwrap(),
            ) {
                Ok(b) => b,
                Err(e) => {
                    println!("Failed to allocate buffer! Error: {:?}", e);
                    return Err(GPUTaskRecordingError::BufferAllocationFailure);
                }
            };

            let staging_buffer = match self.allocator.allocate_buffer(
                &self.device_info,
                (binding.data().len() * 4) as u64,
                BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
                "gpu_staging_only_alloc",
                self.device_info.queue_indices.compute_queue.unwrap(),
            ) {
                Ok(b) => b,
                Err(e) => {
                    println!("Failed to allocate buffer! Error: {:?}", e);
                    return Err(GPUTaskRecordingError::BufferAllocationFailure);
                }
            };

            let readback_buffer = if binding.readback_enabled {
                Some(
                    match self.allocator.allocate_buffer(
                        &self.device_info,
                        (binding.data().len() * 4) as u64,
                        BufferUsageFlags::TRANSFER_DST,
                        gpu_allocator::MemoryLocation::CpuToGpu,
                        "gpu_staging_only_alloc",
                        self.device_info.queue_indices.compute_queue.unwrap(),
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            println!("Failed to allocate buffer! Error: {:?}", e);
                            return Err(GPUTaskRecordingError::BufferAllocationFailure);
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

        let descriptor_set_alloc_info = DescriptorSetAllocateInfo {
            s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: pipeline.descriptor_pool,
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
                    println!("Failed to allocate descriptor set! Error: {}", e);
                    return Err(GPUTaskRecordingError::DescriptorSetAllocationFailure);
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
                        .buffer
                        .clone(),
                    offset: 0,
                    range: (binding.data().len() * 4) as u64,
                });
                descriptor_writes.push(WriteDescriptorSet {
                    s_type: StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set[0].clone(),
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
                println!("Failed to allocate command buffer! Error: {}", e);
                return Err(GPUTaskRecordingError::CommandBufferAllocationFailure);
            }
        };

        match command_buffer_util::begin_command_buffer_recording(
            &self.device_info.device,
            command_buffer.clone(),
            false,
        ) {
            Ok(_) => (),
            Err(e) => {
                println!("Failed to begin command buffer recording! Error: {}", e);
                return Err(GPUTaskRecordingError::CommandBufferRecordingStartFailure);
            }
        }

        unsafe {
            self.device_info.device.cmd_bind_pipeline(
                command_buffer.clone(),
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );

            self.device_info.device.cmd_bind_descriptor_sets(
                command_buffer.clone(),
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout.clone(),
                0,
                &[descriptor_set[0]],
                &[],
            );
        }

        Ok(GPUTask {
            command_buffer,
            device_info: self.device_info.clone(),
            buffers: buffer_backing,
            descriptor_set: descriptor_set[0],
            parent_descriptor_pool: pipeline.descriptor_pool,
            allocator: &mut self.allocator,
        })
    }

    pub fn exec_task<'a>(&self, task: &'a GPUTask) -> Option<GPUSyncPrimitive<'a>> {
        let fence = match command_buffer_util::end_and_submit_command_buffer(
            &self.device_info.device,
            task.command_buffer,
            self.device_info.compute_queue.clone(),
        ) {
            Ok(f) => f,
            Err(e) => {
                println!("Failed to submit command buffer! Error: {}", e);
                return None;
            }
        };

        return Some(GPUSyncPrimitive {
            fence,
            parent: task,
        });
    }

    pub fn await_task(&self, sync: &GPUSyncPrimitive, sync_tensors: Vec<&mut Tensor>) {
        unsafe {
            let _ = self
                .device_info
                .device
                .wait_for_fences(&[sync.fence.clone()], true, u64::MAX);

            self.device_info.device.destroy_fence(sync.fence, None);
        }

        sync_tensors.into_iter().for_each(|tensor| unsafe {
            let backing = match sync.parent.buffers.get(&tensor.id) {
                Some(b) => b,
                None => {
                    println!(
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

impl<'a> GPUTask {
    pub fn op_local_sync_device(self, tensors: Vec<&Tensor>) -> Self {
        tensors.iter().for_each(|tensor| unsafe {
            let backing = match self.buffers.get(&tensor.id) {
                Some(b) => b,
                None => {
                    println!(
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
                    tensor.data().len() * 4 as usize,
                );

            self.device_info.device.cmd_copy_buffer(
                self.command_buffer.clone(),
                backing.staging_buffer.buffer.clone(),
                backing.gpu_buffer.buffer.clone(),
                &[BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (tensor.data().len() * 4) as u64,
                }],
            );
        });

        unsafe {
            self.device_info.device.cmd_pipeline_barrier(
                self.command_buffer.clone(),
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
        unsafe {
            self.device_info.device.cmd_dispatch(
                self.command_buffer.clone(),
                work_group.x,
                work_group.y,
                work_group.z,
            );
        }

        self
    }

    pub fn op_device_sync_local(self, tensors: Vec<&Tensor>) -> Self {
        unsafe {
            self.device_info.device.cmd_pipeline_barrier(
                self.command_buffer.clone(),
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
            let backing = match self.buffers.get(&tensor.id) {
                Some(b) => b,
                None => {
                    println!(
                        "Failed to find backing buffer for tensor! This is an internal issue!"
                    );
                    return;
                }
            };

            if backing.readback_buffer.is_none() {
                println!("Tensor has no readback buffer! Did you enable readback on creation?");
                return;
            }

            self.device_info.device.cmd_copy_buffer(
                self.command_buffer.clone(),
                backing.gpu_buffer.buffer.clone(),
                backing.readback_buffer.as_ref().unwrap().buffer.clone(),
                &[BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (tensor.data().len() * 4) as u64,
                }],
            )
        });

        self
    }
}

impl<'a> Drop for GPUTask {
    fn drop(&mut self) {
        unsafe {
            self.device_info.device.free_command_buffers(
                self.device_info.compute_pool.clone(),
                &[self.command_buffer.clone()],
            );

            // Free backing buffers
            self.buffers.iter_mut().for_each(|(_, buffer)| {
                let gpu_alloc = std::mem::take(&mut buffer.gpu_buffer.allocation);
                let _ = (*self.allocator).vulkan_allocator.free(gpu_alloc);
                self.device_info
                    .device
                    .destroy_buffer(buffer.gpu_buffer.buffer, None);

                let stage_alloc = std::mem::take(&mut buffer.staging_buffer.allocation);
                let _ = (*self.allocator).vulkan_allocator.free(stage_alloc);
                self.device_info
                    .device
                    .destroy_buffer(buffer.staging_buffer.buffer, None);

                if buffer.readback_buffer.is_some() {
                    let readback_alloc =
                        std::mem::take(&mut buffer.readback_buffer.as_mut().unwrap().allocation);
                    let _ = (*self.allocator).vulkan_allocator.free(readback_alloc);
                    self.device_info
                        .device
                        .destroy_buffer(buffer.readback_buffer.as_mut().unwrap().buffer, None);
                }
            });

            let _ = self
                .device_info
                .device
                .free_descriptor_sets(self.parent_descriptor_pool, &[self.descriptor_set]);
        }
    }
}
