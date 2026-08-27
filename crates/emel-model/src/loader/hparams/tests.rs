use super::*;
use crate::loader::test_gguf::{BOOL, F32, F64, Fixture, I8, I16, I32, I64, U8, U16, U32, U64};
use allocation_counter::measure;

#[test]
fn neutral_accessors_cover_raw_signed_array_float_and_flags() {
    let array = [0u32.to_le_bytes(), 9u32.to_le_bytes()].concat();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"i32", U16, 7u16.to_le_bytes())
            .scalar(b"negative", I8, vec![255])
            .array(b"first", U32, &5u32.to_le_bytes(), 1)
            .array(b"nonzero", U32, &array, 2)
            .scalar(b"float", F64, 2.5f64.to_le_bytes())
            .array(b"flags", BOOL, &[0, 2], 2)
            .build(),
    );
    let mut a = Accessor::new();
    let mut i = 3;
    a.assign_i32(&mut gguf, b"i32", &mut i).unwrap();
    assert_eq!(i, 7);
    assert_eq!(
        a.assign_i32(&mut gguf, b"negative", &mut i)
            .unwrap_err()
            .kind,
        ErrorKind::Range
    );
    a.assign_i32_or_first_array_value(&mut gguf, b"first", &mut i)
        .unwrap();
    assert_eq!(i, 5);
    a.assign_first_nonzero_i32_from_array(&mut gguf, b"nonzero", &mut i)
        .unwrap();
    assert_eq!(i, 9);
    let mut f = 0.0;
    a.assign_f32(&mut gguf, b"float", &mut f).unwrap();
    assert!((f - 2.5).abs() < f32::EPSILON);
    let mut flags = [9; 2];
    assert_eq!(
        a.copy_flag_array(&mut gguf, b"flags", &mut flags).unwrap(),
        2
    );
    assert_eq!(flags, [0, 1]);
}

#[test]
fn neutral_accessors_distinguish_missing_wrong_kind_and_preserve_optional() {
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"wrong", I32, 1i32.to_le_bytes())
            .string(b"text", b"x")
            .build(),
    );
    let mut a = Accessor::new();
    let mut value = 11;
    a.assign_i32(&mut gguf, b"missing", &mut value).unwrap();
    assert_eq!(value, 11);
    assert_eq!(
        a.assign_i32(&mut gguf, b"text", &mut value)
            .unwrap_err()
            .kind,
        ErrorKind::WrongKind
    );
    assert_eq!(
        a.assign_first_nonzero_i32_from_array(&mut gguf, b"missing", &mut value)
            .unwrap_err()
            .kind,
        ErrorKind::Missing
    );
}

#[test]
fn neutral_accessors_cover_integer_width_edges_counts_and_capacity() {
    let zeros = [0_u16.to_le_bytes(), 0_u16.to_le_bytes()].concat();
    let oversized_flags = [1_u64.to_le_bytes(), 0_u64.to_le_bytes()].concat();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"u8", U8, [u8::MAX])
            .scalar(b"u16", U16, u16::MAX.to_le_bytes())
            .scalar(b"i16_raw", I16, (-1_i16).to_le_bytes())
            .scalar(b"u32_max_i32", U32, (i32::MAX as u32).to_le_bytes())
            .scalar(b"i32_negative", I32, (-1_i32).to_le_bytes())
            .scalar(b"u64_range", U64, u64::MAX.to_le_bytes())
            .scalar(b"i64_range", I64, i64::MAX.to_le_bytes())
            .scalar(b"f32", F32, 1.25_f32.to_le_bytes())
            .array(b"empty", U8, &[], 0)
            .array(b"zeros", U16, &zeros, 2)
            .array(b"too_many_flags", U64, &oversized_flags, 2)
            .array(b"too_many_bools", BOOL, &[1, 0], 2)
            .build(),
    );
    let mut accessor = Accessor::new();
    let mut value = 0;

    for (key, expected) in [
        (b"u8" as &[u8], i32::from(u8::MAX)),
        (b"u16", i32::from(u16::MAX)),
        (b"i16_raw", i32::from(u16::MAX)),
        (b"u32_max_i32", i32::MAX),
    ] {
        accessor.assign_i32(&mut gguf, key, &mut value).unwrap();
        assert_eq!(value, expected);
    }
    for key in [b"i32_negative" as &[u8], b"u64_range", b"i64_range"] {
        assert_eq!(
            accessor
                .assign_i32(&mut gguf, key, &mut value)
                .unwrap_err()
                .kind,
            ErrorKind::Range
        );
    }
    assert_eq!(
        accessor
            .assign_first_nonzero_i32_from_array(&mut gguf, b"empty", &mut value)
            .unwrap_err()
            .kind,
        ErrorKind::Count
    );
    assert_eq!(
        accessor
            .assign_first_nonzero_i32_from_array(&mut gguf, b"zeros", &mut value)
            .unwrap_err()
            .kind,
        ErrorKind::Count
    );
    let mut unchanged_flags = [9; 1];
    assert_eq!(
        accessor
            .copy_flag_array(&mut gguf, b"too_many_flags", &mut unchanged_flags)
            .unwrap_err()
            .kind,
        ErrorKind::Capacity
    );
    assert_eq!(unchanged_flags, [9]);
    assert_eq!(
        accessor
            .copy_flag_array(&mut gguf, b"too_many_bools", &mut unchanged_flags)
            .unwrap_err()
            .kind,
        ErrorKind::Capacity
    );
    assert_eq!(unchanged_flags, [9]);

    let mut float = 0.0;
    let allocation = measure(|| {
        accessor.assign_f32(&mut gguf, b"f32", &mut float).unwrap();
        accessor.assign_i32(&mut gguf, b"u16", &mut value).unwrap();
    });
    assert_eq!(allocation.count_total, 0);
    assert!((float - 1.25).abs() < f32::EPSILON);
}

#[test]
fn neutral_accessors_cover_optional_and_required_error_routes() {
    let one = 1_u32.to_le_bytes();
    let too_large = u64::MAX.to_le_bytes();
    let integer_flags = [0_u16.to_le_bytes(), 2_u16.to_le_bytes()].concat();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"text", b"wrong")
            .array(b"empty", U32, &[], 0)
            .array(b"one", U32, &one, 1)
            .array(b"range", U64, &too_large, 1)
            .array(b"integer_flags", U16, &integer_flags, 2)
            .build(),
    );
    let mut accessor = Accessor::default();
    let mut integer = 17;
    let mut float = 3.5;

    accessor
        .assign_i32_or_first_array_value(&mut gguf, b"missing", &mut integer)
        .unwrap();
    assert_eq!(integer, 17);
    assert_eq!(
        accessor
            .assign_i32_or_first_array_value(&mut gguf, b"empty", &mut integer)
            .unwrap_err()
            .kind,
        ErrorKind::Count
    );
    assert_eq!(
        accessor
            .assign_i32_or_first_array_value(&mut gguf, b"text", &mut integer)
            .unwrap_err()
            .kind,
        ErrorKind::WrongKind
    );
    assert_eq!(
        accessor
            .assign_first_nonzero_i32_from_array(&mut gguf, b"range", &mut integer)
            .unwrap_err()
            .kind,
        ErrorKind::Range
    );
    assert_eq!(
        accessor
            .assign_first_nonzero_i32_from_array(&mut gguf, b"text", &mut integer)
            .unwrap_err()
            .kind,
        ErrorKind::WrongKind
    );
    accessor
        .assign_first_nonzero_i32_from_array(&mut gguf, b"one", &mut integer)
        .unwrap();
    assert_eq!(integer, 1);

    accessor
        .assign_f32(&mut gguf, b"missing", &mut float)
        .unwrap();
    assert!((float - 3.5).abs() < f32::EPSILON);
    assert_eq!(
        accessor
            .assign_f32(&mut gguf, b"text", &mut float)
            .unwrap_err()
            .kind,
        ErrorKind::WrongKind
    );

    let mut flags = [0; 2];
    assert_eq!(
        accessor
            .copy_flag_array(&mut gguf, b"integer_flags", &mut flags)
            .unwrap(),
        2
    );
    assert_eq!(flags, [0, 1]);
    assert_eq!(
        accessor
            .copy_flag_array(&mut gguf, b"missing", &mut flags)
            .unwrap_err()
            .kind,
        ErrorKind::Missing
    );
    assert_eq!(
        accessor
            .copy_flag_array(&mut gguf, b"text", &mut flags)
            .unwrap_err()
            .kind,
        ErrorKind::WrongKind
    );
}

#[test]
fn neutral_accessors_propagate_unparsed_query_failures() {
    let mut gguf = Loader::new();
    let mut accessor = Accessor::new();
    let mut integer = 0;
    let mut float = 0.0;
    let mut flags = [0; 1];

    for error in [
        accessor
            .assign_i32(&mut gguf, b"key", &mut integer)
            .unwrap_err(),
        accessor
            .assign_i32_or_first_array_value(&mut gguf, b"key", &mut integer)
            .unwrap_err(),
        accessor
            .assign_first_nonzero_i32_from_array(&mut gguf, b"key", &mut integer)
            .unwrap_err(),
        accessor
            .assign_f32(&mut gguf, b"key", &mut float)
            .unwrap_err(),
        accessor
            .copy_flag_array(&mut gguf, b"key", &mut flags)
            .unwrap_err(),
    ] {
        assert_eq!(error.kind, ErrorKind::Query);
    }
}

#[test]
fn scan_outcome_preserves_the_raw_query_error() {
    for error in [
        QueryError::Malformed,
        QueryError::Range,
        QueryError::TypeMismatch,
        QueryError::IndexOutOfBounds,
        QueryError::NotParsed,
        QueryError::Internal,
    ] {
        assert_eq!(ScanOutcome::Query(error), ScanOutcome::Query(error));
    }
}
