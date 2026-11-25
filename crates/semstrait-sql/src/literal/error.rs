use thiserror::Error;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
pub enum LiteralError {
    #[error("the literal's type is not a valid data type")]
    InvalidType,
    #[error("the value is not text of its type")]
    Malformed,
    #[error("the value has more digits than the greatest decimal precision")]
    TooManyDigits,
    #[error("its type would cut a digit of the value")]
    CutsADigit,
    #[error("the value is outside the range of its type")]
    OutOfRange,
    #[error("the value is longer than its type's length")]
    TooLong,
    #[error("a timestamp holds no zone")]
    Zoned,
    #[error("an array value's type is not the array's element")]
    ElementType,
    #[error("the value has no reading as the type")]
    NoReading,
}
