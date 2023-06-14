use std::ptr;

use ash::{
    prelude::VkResult,
    vk::{
        CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsageFlags, CommandPool, Fence, FenceCreateFlags, FenceCreateInfo, Queue,
        StructureType, SubmitInfo,
    },
    Device,
};

pub fn allocate_command_buffer(device: &Device, pool: CommandPool) -> VkResult<CommandBuffer> {
    let command_buffer_allocation_info = CommandBufferAllocateInfo {
        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool: pool,
        level: CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
    };

    unsafe {
        match device.allocate_command_buffers(&command_buffer_allocation_info) {
            Ok(c) => Ok(c[0]),
            Err(e) => Err(e),
        }
    }
}

pub fn begin_command_buffer_recording(
    device: &Device,
    command_buffer: CommandBuffer,
    single_use: bool,
) -> VkResult<()> {
    let begin_info = CommandBufferBeginInfo {
        s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: if single_use {
            CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            CommandBufferUsageFlags::empty()
        },
        p_inheritance_info: ptr::null(),
    };

    unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
}

pub fn end_and_submit_command_buffer(
    device: &Device,
    command_buffer: CommandBuffer,
    dst_queue: Queue,
) -> VkResult<Fence> {
    unsafe {
        device.end_command_buffer(command_buffer)?;

        let submit_info = SubmitInfo {
            s_type: StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };

        let fence_create_info = FenceCreateInfo {
            s_type: StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: FenceCreateFlags::empty(),
        };

        let fence = device.create_fence(&fence_create_info, None)?;

        match device.queue_submit(dst_queue, &[submit_info], fence.clone()) {
            Ok(_) => Ok(fence),
            Err(e) => {
                device.destroy_fence(fence.clone(), None);
                Err(e)
            }
        }
    }
}
