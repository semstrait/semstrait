pub mod local_fs;
mod object_store_fs;

#[cfg(feature = "io-aws")]
pub mod s3_fs;
