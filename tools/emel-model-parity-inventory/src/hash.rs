use sha2::{Digest, Sha256};

pub fn sha256(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}

pub fn semantic_hash(fields: &[&str]) -> String {
    let mut hash = Sha256::new();
    for field in fields {
        let bytes = field.as_bytes();
        hash.update(
            u64::try_from(bytes.len())
                .expect("field length fits u64")
                .to_be_bytes(),
        );
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}
