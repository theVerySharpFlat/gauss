use std::sync::Arc;

use gauss::{compute_init, AllocatorLogConfig, LogConfig, ValidationLayerLogConfig, WorkGroupSize};
use indoc::indoc;
use ndarray::prelude::*;

pub fn main() {
    let compute_manager = compute_init(LogConfig {
        validation_config: Some(ValidationLayerLogConfig {
            log_errors: true,
            log_warnings: true,
            log_verbose_info: true,
        }),
        allocator_config: Some(AllocatorLogConfig {
            log_memory_information: true,
            log_leaks_on_shutdown: true,
            store_stack_traces: false,
            log_allocations: false,
            log_frees: false,
            log_stack_traces: false,
        }),
    })
    .unwrap();

    let shader = indoc! {"
        #version 450
        
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout(set = 0, binding = 0) buffer buf_in  {  float in_a[];  };
        layout(set = 0, binding = 1) buffer buf_out {  float out_a[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] = in_a[index] * in_a[index];
        }
    "};

    let tensor_in = compute_manager.create_tensor(array![1.0, 2.0, 3.0, 4.0, 5.0], false);
    let mut tensor_out = compute_manager.create_tensor(array![5.0, 4.0, 3.0, 2.0, 1.0], true);

    let pipeline = compute_manager
        .clone()
        .build_pipeline(
            compute_manager
                .clone()
                .compile_program(shader, "basic_compute", true)
                .unwrap(),
            2,
        )
        .unwrap();

    let task = compute_manager
        .clone()
        .new_task(&pipeline, vec![&tensor_in, &tensor_out])
        .op_local_sync_device(vec![&tensor_in, &tensor_out])
        .op_pipeline_dispatch(WorkGroupSize { x: 5, y: 1, z: 1 })
        .op_device_sync_local(vec![&tensor_out])
        .finalize()
        .unwrap();

    let running_task = compute_manager.exec_task(&task).unwrap();

    log::trace!("Strong RefCount: {}", Arc::strong_count(&compute_manager));
    log::trace!("Weak RefCount: {}", Arc::weak_count(&compute_manager));

    compute_manager.await_task(&running_task, vec![&mut tensor_out]);
    println!("Data: {}", tensor_out.data());
}
