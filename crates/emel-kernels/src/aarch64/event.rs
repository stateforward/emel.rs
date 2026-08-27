//! `AArch64` kernel events.

pub use super::sm::{
    Aarch64Generic, BinaryAdd, BinaryDiv, BinaryMul, BinarySub, BroadcastAdd, BroadcastMul,
    ConvTranspose1d, Dup, Gemv, KernelEvent, MulMatArgmaxQ6Packed, MulMatF32, MulMatQ4_0Vector,
    MulMatQ4_1Vector, MulMatQ4K, MulMatQ4KVector, MulMatQ4PackedBl4, MulMatQ4PackedBl4MatrixX4,
    MulMatQ4PackedBl8, MulMatQ4PackedBl8MatrixX4, MulMatQ4PackedBl8MatrixX8, MulMatQ4PackedF32Bl4,
    MulMatQ4PackedF32Bl8, MulMatQ5_0Vector, MulMatQ6Packed, MulMatQ6PackedMatrixX4,
    MulMatQ6Prepared, MulMatQ6PreparedMatrixX4, MulMatQ6PreparedMatrixX8, MulMatQ8_0PackedBl4,
    MulMatQ8_0PackedBl8, MulMatQ8_0PackedBl8MatrixX4, MulMatQ8_0Vector, PowerSqr, PowerSqrt,
    UnaryAbs, UnaryNeg, UnaryRelu, UnarySilu,
};
