#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("image: {0}")]
    Image(String),
    #[error("audio: {0}")]
    Audio(String),
    #[error("text: {0}")]
    Text(String),
    #[error("config: {0}")]
    Config(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<::image::ImageError> for Error {
    fn from(e: ::image::ImageError) -> Self {
        Error::Image(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Config(e.to_string())
    }
}
