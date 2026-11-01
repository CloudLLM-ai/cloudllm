//! Known-answer tests pinning the SHA-256 behaviour that bearer-token
//! authentication relies on.
//!
//! `static_token_matches` compares `Sha256::digest` outputs with
//! `subtle::ConstantTimeEq`. These vectors (FIPS 180-4 / NIST examples) fail
//! loudly if a `sha2`/`digest` upgrade ever changes the digest bytes, the
//! output length, or the slice-comparison behaviour.

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256("abc") from FIPS 180-4.
#[test]
fn test_sha256_known_answer_abc() {
    assert_eq!(
        hex(&Sha256::digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

/// SHA-256("") from FIPS 180-4.
#[test]
fn test_sha256_known_answer_empty() {
    assert_eq!(
        hex(&Sha256::digest(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

/// The 448-bit NIST message exercises the two-block padding path.
#[test]
fn test_sha256_known_answer_two_block_message() {
    assert_eq!(
        hex(&Sha256::digest(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

/// Digests are 32 bytes and the constant-time comparison distinguishes equal
/// from different inputs, as the auth helpers expect.
#[test]
fn test_digest_ct_eq_equal_and_different() {
    let a = Sha256::digest(b"secret-token");
    let b = Sha256::digest(b"secret-token");
    let c = Sha256::digest(b"secret-tokeN");
    assert_eq!(a.len(), 32);
    assert!(bool::from(a.ct_eq(&b)));
    assert!(!bool::from(a.ct_eq(&c)));
}
