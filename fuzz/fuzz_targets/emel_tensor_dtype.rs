#![no_main]

use emel_tensor::dtype::{ConversionError, SerializedType};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Copy)]
struct ExpectedLayout {
    code: u32,
    columns: u16,
    rows: u8,
    bytes: u16,
    minimum_dimensions: u8,
}

const EXPECTED: [ExpectedLayout; 34] = [
    layout(0, 1, 1, 4),
    layout(1, 1, 1, 2),
    layout(2, 32, 1, 18),
    layout(3, 32, 1, 20),
    layout(6, 32, 1, 22),
    layout(7, 32, 1, 24),
    layout(8, 32, 1, 34),
    layout(9, 32, 1, 36),
    layout(10, 256, 1, 84),
    layout(11, 256, 1, 110),
    layout(12, 256, 1, 144),
    layout(13, 256, 1, 176),
    layout(14, 256, 1, 210),
    layout(15, 256, 1, 292),
    layout(16, 256, 1, 66),
    layout(17, 256, 1, 74),
    layout(18, 256, 1, 98),
    layout(19, 256, 1, 50),
    layout(20, 32, 1, 18),
    layout(21, 256, 1, 110),
    layout(22, 256, 1, 82),
    layout(23, 256, 1, 136),
    layout(24, 1, 1, 1),
    layout(25, 1, 1, 2),
    layout(26, 1, 1, 4),
    layout(27, 1, 1, 8),
    layout(28, 1, 1, 8),
    layout(29, 256, 1, 56),
    layout(30, 1, 1, 2),
    layout(34, 256, 1, 54),
    layout(35, 256, 1, 66),
    layout(39, 32, 1, 17),
    packed(41),
    packed(42),
];

const REJECTED_CODES: [u32; 15] = [
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
];

const fn layout(code: u32, columns: u16, rows: u8, bytes: u16) -> ExpectedLayout {
    ExpectedLayout {
        code,
        columns,
        rows,
        bytes,
        minimum_dimensions: 0,
    }
}

const fn packed(code: u32) -> ExpectedLayout {
    ExpectedLayout {
        code,
        columns: 256,
        rows: 8,
        bytes: 1152,
        minimum_dimensions: 2,
    }
}

fn expected_layout(code: u32) -> Option<ExpectedLayout> {
    EXPECTED.iter().copied().find(|layout| layout.code == code)
}

fn expected_size(
    layout: ExpectedLayout,
    dimensions: [u64; 4],
    dimension_count: u32,
) -> Result<u64, ConversionError> {
    let count = usize::try_from(dimension_count).map_err(|_| ConversionError::InvalidShape)?;
    let active = dimensions
        .get(..count)
        .ok_or(ConversionError::InvalidShape)?;
    if dimension_count < u32::from(layout.minimum_dimensions) {
        return Err(ConversionError::InvalidShape);
    }
    let columns = active.first().copied().unwrap_or(1);
    let columns_per_group = u64::from(layout.columns);
    if !columns.is_multiple_of(columns_per_group) {
        return Err(ConversionError::InvalidShape);
    }
    let rows = active
        .get(1..)
        .unwrap_or_default()
        .iter()
        .try_fold(1_u64, |rows, dimension| {
            rows.checked_mul(*dimension)
                .ok_or(ConversionError::Capacity)
        })?;
    let rows_per_group = u64::from(layout.rows);
    let row_groups = rows.div_ceil(rows_per_group);
    (columns / columns_per_group)
        .checked_mul(row_groups)
        .and_then(|groups| groups.checked_mul(u64::from(layout.bytes)))
        .ok_or(ConversionError::Capacity)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 38 {
        return;
    }
    let raw_code = u32::from_le_bytes(data[1..5].try_into().unwrap());
    let code = match data[0] % 3 {
        0 => EXPECTED[usize::from(data[1]) % EXPECTED.len()].code,
        1 => REJECTED_CODES[usize::from(data[1]) % REJECTED_CODES.len()],
        _ => raw_code,
    };
    let dimension_count = u32::from(data[5]);
    let mut dimensions = [0_u64; 4];
    for (index, dimension) in dimensions.iter_mut().enumerate() {
        let start = 6 + index * 8;
        *dimension = u64::from_le_bytes(data[start..start + 8].try_into().unwrap());
    }

    match expected_layout(code) {
        Some(layout) => {
            let tensor_type = SerializedType::try_from(code).expect("source-owned accepted code");
            assert_eq!(tensor_type.wire_code(), code);
            assert_eq!(
                tensor_type.data_size(dimensions, dimension_count),
                expected_size(layout, dimensions, dimension_count)
            );
        }
        None => assert_eq!(
            SerializedType::try_from(code),
            Err(ConversionError::UnknownSerializedType(code))
        ),
    }

    assert_eq!(SerializedType::Q4K.data_size([256, 9, 1, 1], 2), Ok(1_296));
    for packed in [SerializedType::Q4Kx8Bl4, SerializedType::Q4Kx8Bl8] {
        assert_eq!(packed.data_size([256, 1, 1, 1], 2), Ok(1_152));
        assert_eq!(packed.data_size([256, 8, 1, 1], 2), Ok(1_152));
        assert_eq!(packed.data_size([256, 9, 1, 1], 2), Ok(2_304));
    }
});
