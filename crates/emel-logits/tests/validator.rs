#![allow(clippy::float_cmp, missing_docs)]

use emel_logits::validator::{Build, BuildResult, Validator, ValidatorError};
use sml as _;

#[test]
fn build_normalizes_logits() {
    let logits = [1.0, 4.0, 2.5];
    let mut ids = [-1; 3];
    let mut scores = [99.0; 3];
    let mut count = -1;
    let mut error = ValidatorError::InternalError;
    let result = Validator::new().process_event(Build::new(
        &logits,
        3,
        &mut ids,
        &mut scores,
        3,
        &mut count,
        &mut error,
    ));
    assert_eq!(result, Ok(BuildResult { candidate_count: 3 }));
    assert_eq!(ids, [0, 1, 2]);
    assert_eq!(scores, [-3.0, 0.0, -1.5]);
    assert_eq!(count, 3);
    assert_eq!(error, ValidatorError::None);
}

#[test]
fn build_rejects_invalid_sizes_and_resets_outputs() {
    let logits = [1.0, 2.0];
    let cases = [(0, 2), (2, 1), (-1, 2), (3, 3)];
    for (vocab_size, capacity) in cases {
        let mut ids = [7; 2];
        let mut scores = [8.0; 2];
        let mut count = 9;
        let mut error = ValidatorError::BackendError;
        let result = Validator::new().process_event(Build::new(
            &logits,
            vocab_size,
            &mut ids,
            &mut scores,
            capacity,
            &mut count,
            &mut error,
        ));
        assert_eq!(result, Err(ValidatorError::InvalidRequest));
        assert_eq!(count, 0);
        assert_eq!(error, ValidatorError::InvalidRequest);
    }
}
