#![allow(missing_docs)]

use sml as _;

use emel_tensor::dtype::{ConversionError, KernelType, SerializedLayoutKind, SerializedType};

const ORDINARY: &[(u32, SerializedType, u16, u16)] = &[
    (0, SerializedType::F32, 1, 4),
    (1, SerializedType::F16, 1, 2),
    (2, SerializedType::Q4_0, 32, 18),
    (3, SerializedType::Q4_1, 32, 20),
    (6, SerializedType::Q5_0, 32, 22),
    (7, SerializedType::Q5_1, 32, 24),
    (8, SerializedType::Q8_0, 32, 34),
    (9, SerializedType::Q8_1, 32, 36),
    (10, SerializedType::Q2K, 256, 84),
    (11, SerializedType::Q3K, 256, 110),
    (12, SerializedType::Q4K, 256, 144),
    (13, SerializedType::Q5K, 256, 176),
    (14, SerializedType::Q6K, 256, 210),
    (15, SerializedType::Q8K, 256, 292),
    (16, SerializedType::Iq2Xxs, 256, 66),
    (17, SerializedType::Iq2Xs, 256, 74),
    (18, SerializedType::Iq3Xxs, 256, 98),
    (19, SerializedType::Iq1S, 256, 50),
    (20, SerializedType::Iq4Nl, 32, 18),
    (21, SerializedType::Iq3S, 256, 110),
    (22, SerializedType::Iq2S, 256, 82),
    (23, SerializedType::Iq4Xs, 256, 136),
    (24, SerializedType::I8, 1, 1),
    (25, SerializedType::I16, 1, 2),
    (26, SerializedType::I32, 1, 4),
    (27, SerializedType::I64, 1, 8),
    (28, SerializedType::F64, 1, 8),
    (29, SerializedType::Iq1M, 256, 56),
    (30, SerializedType::Bf16, 1, 2),
    (34, SerializedType::Tq1_0, 256, 54),
    (35, SerializedType::Tq2_0, 256, 66),
    (39, SerializedType::Mxfp4, 32, 17),
];

const KERNEL_EQUIVALENTS: &[(SerializedType, KernelType)] = &[
    (SerializedType::F32, KernelType::F32),
    (SerializedType::F16, KernelType::F16),
    (SerializedType::Q4_0, KernelType::Q4_0),
    (SerializedType::Q4_1, KernelType::Q4_1),
    (SerializedType::Q5_0, KernelType::Q5_0),
    (SerializedType::Q5_1, KernelType::Q5_1),
    (SerializedType::Q8_0, KernelType::Q8_0),
    (SerializedType::Q8_1, KernelType::Q8_1),
    (SerializedType::Q2K, KernelType::Q2K),
    (SerializedType::Q3K, KernelType::Q3K),
    (SerializedType::Q4K, KernelType::Q4K),
    (SerializedType::Q5K, KernelType::Q5K),
    (SerializedType::Q6K, KernelType::Q6K),
    (SerializedType::Q8K, KernelType::Q8K),
    (SerializedType::Iq2Xxs, KernelType::Iq2Xxs),
    (SerializedType::Iq2Xs, KernelType::Iq2Xs),
    (SerializedType::Iq3Xxs, KernelType::Iq3Xxs),
    (SerializedType::Iq1S, KernelType::Iq1S),
    (SerializedType::Iq4Nl, KernelType::Iq4Nl),
    (SerializedType::Iq3S, KernelType::Iq3S),
    (SerializedType::Iq2S, KernelType::Iq2S),
    (SerializedType::Iq4Xs, KernelType::Iq4Xs),
    (SerializedType::I8, KernelType::I8),
    (SerializedType::I16, KernelType::I16),
    (SerializedType::I32, KernelType::I32),
    (SerializedType::I64, KernelType::I64),
    (SerializedType::F64, KernelType::F64),
    (SerializedType::Iq1M, KernelType::Iq1M),
    (SerializedType::Bf16, KernelType::Bf16),
    (SerializedType::Tq1_0, KernelType::Tq1_0),
    (SerializedType::Tq2_0, KernelType::Tq2_0),
    (SerializedType::Q4Kx8Bl4, KernelType::Q4Kx8Bl4),
    (SerializedType::Q4Kx8Bl8, KernelType::Q4Kx8Bl8),
];

#[test]
fn every_ordinary_wire_code_has_the_exact_layout_and_size() {
    assert_eq!(ORDINARY.len(), 32, "source-backed ordinary type count");
    assert_eq!(ORDINARY.len() + 2, 34, "ordinary plus packed type count");
    for &(code, serialized, elements, bytes) in ORDINARY {
        assert_eq!(SerializedType::try_from(code), Ok(serialized));
        assert_eq!(serialized.wire_code(), code);
        let layout = serialized.layout();
        assert_eq!(layout.kind(), SerializedLayoutKind::Block);
        assert_eq!(layout.columns(), elements);
        assert_eq!(layout.rows(), 1);
        assert_eq!(layout.bytes(), bytes);
        let dimensions = [u64::from(elements) * 3, 2, 1, 1];
        assert_eq!(
            serialized.data_size(dimensions, 2),
            Ok(u64::from(bytes) * 6)
        );
    }
}

#[test]
fn ordinary_zero_dimension_and_shape_failures_are_exact() {
    assert_eq!(SerializedType::F32.data_size([1; 4], 0), Ok(4));
    assert_eq!(
        SerializedType::Q4K.data_size([255, 1, 1, 1], 1),
        Err(ConversionError::InvalidShape)
    );
    assert_eq!(
        SerializedType::F32.data_size([1; 4], 5),
        Err(ConversionError::InvalidShape)
    );
    assert_eq!(
        SerializedType::F32.data_size([u64::MAX, 2, 1, 1], 2),
        Err(ConversionError::Capacity)
    );
    assert_eq!(
        SerializedType::F64.data_size([u64::MAX, 1, 1, 1], 1),
        Err(ConversionError::Capacity)
    );
}

#[test]
fn packed_q4_k_x8_layout_handles_partial_groups_and_overflow() {
    for serialized in [SerializedType::Q4Kx8Bl4, SerializedType::Q4Kx8Bl8] {
        let layout = serialized.layout();
        assert_eq!(layout.kind(), SerializedLayoutKind::PackedQ4Kx8);
        assert_eq!(layout.columns(), 256);
        assert_eq!(layout.rows(), 8);
        assert_eq!(layout.bytes(), 1152);
        assert_eq!(serialized.data_size([256, 1, 1, 1], 2), Ok(1152));
        assert_eq!(serialized.data_size([512, 9, 2, 1], 3), Ok(6912));
        for (dimensions, count) in [
            ([256, 1, 1, 1], 0),
            ([256, 1, 1, 1], 1),
            ([256, 1, 1, 1], 5),
            ([255, 8, 1, 1], 2),
        ] {
            assert_eq!(
                serialized.data_size(dimensions, count),
                Err(ConversionError::InvalidShape)
            );
        }
        assert_eq!(
            serialized.data_size([256, u64::MAX, 2, 1], 3),
            Err(ConversionError::Capacity)
        );
        assert_eq!(
            serialized.data_size([u64::MAX - 255, u64::MAX, 1, 1], 2),
            Err(ConversionError::Capacity)
        );
    }
}

#[test]
fn rejected_codes_never_truncate_or_enter_the_serialized_domain() {
    for code in [
        4,
        5,
        31,
        32,
        33,
        36,
        37,
        38,
        40,
        43,
        44,
        45,
        255,
        256,
        u32::MAX,
    ] {
        assert_eq!(
            SerializedType::try_from(code),
            Err(ConversionError::UnknownSerializedType(code))
        );
    }
    for code in 45..=512 {
        assert!(SerializedType::try_from(code).is_err());
    }
}

#[test]
fn serialized_39_never_aliases_the_internal_kernel_39() {
    assert_eq!(SerializedType::try_from(39), Ok(SerializedType::Mxfp4));
    assert_eq!(
        KernelType::try_from(SerializedType::Mxfp4),
        Err(ConversionError::NoKernelEquivalent(SerializedType::Mxfp4))
    );
    assert_eq!(
        KernelType::try_from(SerializedType::Q4Kx8Bl4),
        Ok(KernelType::Q4Kx8Bl4)
    );
    assert_eq!(
        KernelType::try_from(SerializedType::Q4Kx8Bl8),
        Ok(KernelType::Q4Kx8Bl8)
    );
    assert_eq!(KERNEL_EQUIVALENTS.len(), 33);
    for &(serialized, kernel) in KERNEL_EQUIVALENTS {
        assert_eq!(KernelType::try_from(serialized), Ok(kernel));
    }
}
