use self::init_error::InitError;

mod init_error;
mod device;
mod instance;

#[derive(Debug)]
pub struct ComputeManager {

}

pub fn compute_init() -> Result<ComputeManager, InitError> {
    Ok(ComputeManager {  })
}
