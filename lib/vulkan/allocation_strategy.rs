use std::ptr;

use ash::vk;
use ash::vk::{BufferCreateFlags, BufferCreateInfo, BufferUsageFlags, SharingMode, StructureType};

use gpu_allocator::vulkan::{Allocation, AllocationScheme};
use gpu_allocator::MemoryLocation;
use gpu_allocator::{
    vulkan::{AllocationCreateDesc, Allocator as VulkanAllocator, AllocatorCreateDesc},
    AllocatorDebugSettings,
};

use ndarray::prelude::*;

use super::ComputeManager;
use super::{device::DeviceInfo, instance::InstanceInfo};

pub struct Allocator {
    pub(super) vulkan_allocator: VulkanAllocator,
    device: ash::Device,
}

pub struct AllocatorBufferCreateInfo {
    size: u64,
}

pub struct Buffer {
    pub(super) buffer: vk::Buffer,
    pub(super) allocation: Allocation,
}

pub struct Tensor {
    pub(super) id: u32,
    pub(super) readback_enabled: bool,

    local_data: Array<f32, Ix1>,
}

pub struct NetBufferAllocation {
    gpu_buffers: Vec<Buffer>,
    staging_buffers: Vec<Buffer>,
}

#[derive(Debug, Clone, Copy)]
pub enum AllocationError {
    AllocatorCreationFailure,
    BufferCreationFailure,
    MemoryAllocationError,
    MemoryMapFailure,
    MemoryBindFailure,
    TransferFailure,
}

impl ComputeManager {
    pub fn create_tensor(&mut self, data: Array<f32, Ix1>, enable_readback: bool) -> Tensor {
        Tensor {
            id: {
                self.current_tensor_id += 1;
                self.current_tensor_id - 1
            },
            readback_enabled: enable_readback,
            local_data: data,
        }
    }
}

impl Tensor {
    pub fn data(&self) -> &Array<f32, Ix1> {
        &self.local_data
    }

    pub fn data_mut(&mut self) -> &mut Array<f32, Ix1> {
        &mut self.local_data
    }
}

impl Allocator {
    pub fn new(
        instance_info: &InstanceInfo,
        device_info: &DeviceInfo,
    ) -> Result<Self, AllocationError> {
        let vulkan_allocator = match VulkanAllocator::new(&AllocatorCreateDesc {
            instance: instance_info.instance.clone(),
            device: device_info.device.clone(),
            physical_device: device_info.physical_device,
            debug_settings: AllocatorDebugSettings {
                log_memory_information: true,
                log_leaks_on_shutdown: true,
                store_stack_traces: false,
                log_allocations: true,
                log_frees: false,
                log_stack_traces: false,
            },
            buffer_device_address: false,
        }) {
            Ok(a) => a,
            Err(e) => {
                println!("Failed to create allocator! Error: \"{}\"", e);
                return Err(AllocationError::AllocatorCreationFailure);
            }
        };

        Ok(Allocator {
            vulkan_allocator,
            device: device_info.device.clone(),
        })
    }

    fn free_buffer(self: &mut Self, buffer: Buffer) {
        let _ = self.vulkan_allocator.free(buffer.allocation);
        unsafe { self.device.destroy_buffer(buffer.buffer, None) }
    }

    pub fn allocate_buffer(
        &mut self,
        device_info: &DeviceInfo,
        size: u64,
        usage: BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
        queue_family: u32,
    ) -> Result<Buffer, AllocationError> {
        let queue_families = [queue_family];

        let buffer_create_info = BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: BufferCreateFlags::empty(),
            size,
            usage,
            sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: queue_families.as_ptr(),
        };

        let buffer = unsafe {
            match device_info.device.create_buffer(&buffer_create_info, None) {
                Ok(b) => b,
                Err(e) => {
                    println!("Failed to allocate buffer with error {}", e);
                    return Err(AllocationError::BufferCreationFailure);
                }
            }
        };

        let buffer_memory_requirements = unsafe {
            device_info
                .device
                .get_buffer_memory_requirements(buffer.clone())
        };

        let buffer_allocation = match self.vulkan_allocator.allocate(&AllocationCreateDesc {
            name,
            requirements: buffer_memory_requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                println!("Failed to allocate backing memory for buffer! Error: {}", e);
                return Err(AllocationError::MemoryAllocationError);
            }
        };

        unsafe {
            match device_info.device.bind_buffer_memory(
                buffer,
                buffer_allocation.memory(),
                buffer_allocation.offset(),
            ) {
                Ok(_) => (),
                Err(e) => {
                    println!("Failed to bind buffer memory! Error: {}", e);
                    return Err(AllocationError::MemoryBindFailure);
                }
            };
        }

        Ok(Buffer {
            buffer,
            allocation: buffer_allocation,
        })
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            (*self.parent)
                .device_info
                .device
                .destroy_buffer(self.gpu_buffer.buffer, None);
            (*self.parent)
                .device_info
                .device
                .destroy_buffer(self.staging_buffer.buffer, None);

            if self.readback_buffer.is_some() {
                (*self.parent)
                    .device_info
                    .device
                    .destroy_buffer(self.readback_buffer.as_ref().unwrap().buffer, None);

                let _ = (*self.parent)
                    .allocator
                    .vulkan_allocator
                    .free(std::mem::take(
                        &mut self.readback_buffer.as_mut().unwrap().allocation,
                    ));
            }

            let _ = (*self.parent)
                .allocator
                .vulkan_allocator
                .free(std::mem::take(&mut self.staging_buffer.allocation));
            let _ = (*self.parent)
                .allocator
                .vulkan_allocator
                .free(std::mem::take(&mut self.gpu_buffer.allocation));

            println!("hereeeee");
        }
    }
}
