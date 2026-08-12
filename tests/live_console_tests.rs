//! Isolated tests for the live console harness.

use cloudllm::live_console::{format_duration, utf8_prefix, LiveConsoleHandler};

#[test]
fn preview_truncates() {
    assert_eq!(LiveConsoleHandler::preview("hi", 10), "hi");
    assert_eq!(LiveConsoleHandler::preview("hello world", 5), "hello...");
}

#[test]
fn format_mins() {
    assert_eq!(format_duration(9), "9s");
    assert_eq!(format_duration(75), "1m 15s");
}

#[test]
fn print_env_knobs_does_not_panic() {
    LiveConsoleHandler::print_env_knobs();
}

#[test]
fn embedded_mentisdb_opens_local_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let handle = LiveConsoleHandler::open_embedded_mentisdb_at(
        dir.path().to_path_buf(),
        "cloudllm-test".to_string(),
    )
    .expect("embedded open");
    assert_eq!(handle.chain_key, "cloudllm-test");
    assert!(handle.dir.exists());
}

#[test]
fn embedded_mentisdb_aborts_on_unwritable_parent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let not_a_dir = tmp.path().join("file");
    std::fs::write(&not_a_dir, b"x").expect("file");
    let nested = not_a_dir.join("chain");
    let err = LiveConsoleHandler::open_embedded_mentisdb_at(nested, "x".into());
    assert!(err.is_err());
}

#[test]
fn utf8_prefix_does_not_split_multibyte_chars() {
    let snowman = "☃";
    assert_eq!(utf8_prefix(snowman, 0), "");
    assert_eq!(utf8_prefix(snowman, 1), "");
    assert_eq!(utf8_prefix(snowman, 2), "");
    assert_eq!(utf8_prefix(snowman, 3), snowman);
    assert_eq!(utf8_prefix("ab☃cd", 4), "ab");
    assert_eq!(utf8_prefix("hello", 5), "hello");
    assert_eq!(utf8_prefix("hello", 100), "hello");
}
