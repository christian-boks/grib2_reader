use thiserror::Error;
#[cfg(feature = "async")]
use tokio::io;

#[derive(Error, Debug)]
/// List of errors the library can return when reading a GRIB file
pub enum Grib2Error {
    #[cfg(feature = "async")]
    #[error("IO Error")]
    /// An IO error occurred while handling the supplied file
    IoError(#[from] io::Error),

    #[error("Wrong Grib2 header")]
    /// The header didn't match the expected GRIB header
    WrongHeader,

    #[error("Wrong Grib version. Only version 2 is supported")]
    /// The contained version number didn't match 0x02
    WrongVersion(u8),

    #[error("Tried to decode more data than we have")]
    /// The bitstream representing the data didn't have the expected length
    DataDecodeFailed(String),

    #[error("End of file")]
    /// We read the last of the file
    EOF,
}
