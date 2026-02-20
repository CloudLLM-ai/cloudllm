//! Comprehensive tests for `BashTool` and `BashProtocol`.
//!
//! Organised into eight sections:
//!
//! 1. **Output & exit codes** — basic correctness of stdout/stderr capture and exit-code handling
//! 2. **Timeout enforcement** — actual deadline is applied, not just stored
//! 3. **Allowlist security** — only listed commands can run
//! 4. **Denylist security** — listed commands are blocked; denylist wins over allowlist
//! 5. **Environment variables** — custom env vars are visible inside the shell
//! 6. **Platform** — shell path and default-platform invariants
//! 7. **Thread safety** — shared `Arc<BashTool>` survives concurrent use
//! 8. **BashProtocol (ToolProtocol integration)** — wrapper behaviour, error mapping, metadata

use cloudllm::tool_protocol::ToolProtocol;
use cloudllm::tool_protocols::BashProtocol;
use cloudllm::tools::{BashError, BashTool, Platform};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tempfile;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Detect the host OS so tests that exercise OS-specific behaviour can skip.
fn is_macos() -> bool {
    cfg!(target_os = "macos")
}

// ─── 1. Output & exit codes ───────────────────────────────────────────────────

#[tokio::test]
async fn test_stdout_is_captured() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("echo hello").await.unwrap();
    assert!(r.success);
    assert_eq!(r.stdout.trim(), "hello");
    assert_eq!(r.exit_code, 0);
}

#[tokio::test]
async fn test_stderr_is_captured_separately() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("echo error_text >&2").await.unwrap();
    assert!(r.success, "command should succeed");
    assert!(r.stdout.trim().is_empty(), "nothing on stdout");
    assert!(r.stderr.contains("error_text"), "stderr should have the text");
}

#[tokio::test]
async fn test_stdout_and_stderr_simultaneously() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash
        .execute("echo out_text; echo err_text >&2")
        .await
        .unwrap();
    assert!(r.success);
    assert!(r.stdout.contains("out_text"));
    assert!(r.stderr.contains("err_text"));
}

#[tokio::test]
async fn test_empty_output_command() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("true").await.unwrap();
    assert!(r.success);
    assert!(r.stdout.is_empty());
    assert!(r.stderr.is_empty());
}

#[tokio::test]
async fn test_exit_code_zero_means_success() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("exit 0").await.unwrap();
    assert!(r.success);
    assert_eq!(r.exit_code, 0);
}

#[tokio::test]
async fn test_exit_code_1() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("exit 1").await.unwrap();
    assert!(!r.success);
    assert_eq!(r.exit_code, 1);
}

#[tokio::test]
async fn test_exit_code_42() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("exit 42").await.unwrap();
    assert!(!r.success);
    assert_eq!(r.exit_code, 42);
}

#[tokio::test]
async fn test_exit_code_127_command_not_found() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash
        .execute("this_command_does_not_exist_xyz_abc_123")
        .await
        .unwrap();
    assert!(!r.success);
    assert_eq!(r.exit_code, 127, "shell exit code for 'command not found'");
    // Every POSIX shell emits something about "not found"
    assert!(
        r.stderr.to_lowercase().contains("not found")
            || r.stderr.to_lowercase().contains("no such"),
        "stderr should mention 'not found'; got: {:?}",
        r.stderr
    );
}

#[tokio::test]
async fn test_multiline_output_order_is_preserved() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash
        .execute("printf 'line1\nline2\nline3\n'")
        .await
        .unwrap();
    assert!(r.success);
    let lines: Vec<&str> = r.stdout.trim().lines().collect();
    assert_eq!(lines, vec!["line1", "line2", "line3"]);
}

#[tokio::test]
async fn test_unicode_output() {
    let bash = BashTool::new(Platform::Linux);
    // Three-byte UTF-8 characters (box-drawing) must not panic or corrupt output.
    let r = bash.execute("printf '┌──┐\\n│  │\\n└──┘\\n'").await.unwrap();
    assert!(r.success);
    assert!(r.stdout.contains('┌'));
    assert!(r.stdout.contains('└'));
}

#[tokio::test]
async fn test_duration_is_positive() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("echo x").await.unwrap();
    assert!(r.duration_ms > 0, "duration_ms must be at least 1 ms");
}

#[tokio::test]
async fn test_slower_command_has_larger_duration() {
    let bash = BashTool::new(Platform::Linux);
    let fast = Instant::now();
    bash.execute("echo x").await.unwrap();
    let fast_ms = fast.elapsed().as_millis() as u64;

    let slow_r = bash.execute("sleep 0.05").await.unwrap();
    // The slow command's reported duration should exceed the fast command's wall time.
    assert!(
        slow_r.duration_ms >= fast_ms || slow_r.duration_ms >= 40,
        "slow command duration ({} ms) should be meaningfully longer",
        slow_r.duration_ms
    );
}

#[tokio::test]
async fn test_pipeline_command() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash
        .execute("printf '3\\n1\\n2\\n' | sort -n")
        .await
        .unwrap();
    assert!(r.success);
    let lines: Vec<&str> = r.stdout.trim().lines().collect();
    assert_eq!(lines, ["1", "2", "3"]);
}

#[tokio::test]
async fn test_semicolon_chained_commands() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("echo a; echo b; echo c").await.unwrap();
    assert!(r.success);
    assert!(r.stdout.contains('a'));
    assert!(r.stdout.contains('b'));
    assert!(r.stdout.contains('c'));
}

#[tokio::test]
async fn test_command_substitution() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("echo $(echo inner)").await.unwrap();
    assert!(r.success);
    assert_eq!(r.stdout.trim(), "inner");
}

#[tokio::test]
async fn test_arithmetic_expansion() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("echo $((6 * 7))").await.unwrap();
    assert!(r.success);
    assert_eq!(r.stdout.trim(), "42");
}

#[tokio::test]
async fn test_false_command_produces_failure() {
    let bash = BashTool::new(Platform::Linux);
    let r = bash.execute("false").await.unwrap();
    assert!(!r.success);
    assert_ne!(r.exit_code, 0);
}

// ─── 2. Timeout enforcement ───────────────────────────────────────────────────

#[tokio::test]
async fn test_timeout_default_is_30_seconds() {
    let bash = BashTool::new(Platform::Linux);
    assert_eq!(bash.timeout_secs(), 30);
}

#[tokio::test]
async fn test_timeout_builder_sets_value() {
    let bash = BashTool::new(Platform::Linux).with_timeout(5);
    assert_eq!(bash.timeout_secs(), 5);
}

#[tokio::test]
async fn test_timeout_is_actually_enforced() {
    // A 1-second timeout must abort a 10-second sleep.
    let bash = BashTool::new(Platform::Linux).with_timeout(1);
    let result = bash.execute("sleep 10").await;
    assert!(
        result.is_err(),
        "command that exceeds timeout must return Err"
    );
    match result.unwrap_err() {
        BashError::Timeout(_) => {}
        other => panic!("expected BashError::Timeout, got: {}", other),
    }
}

#[tokio::test]
async fn test_fast_command_finishes_before_tight_timeout() {
    // A quick echo must complete even with a very short (but non-zero) timeout.
    let bash = BashTool::new(Platform::Linux).with_timeout(5);
    let r = bash.execute("echo fast").await.unwrap();
    assert!(r.success);
    assert_eq!(r.stdout.trim(), "fast");
}

// ─── 3. Allowlist security ────────────────────────────────────────────────────

#[tokio::test]
async fn test_allowlist_permits_exact_command() {
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string()]);
    let r = bash.execute("echo ok").await;
    assert!(r.is_ok(), "allowlisted command should be permitted");
    assert!(r.unwrap().success);
}

#[tokio::test]
async fn test_allowlist_permits_prefix_with_args() {
    // "ls" is in the allowlist; "ls -la /tmp" starts with "ls " so must be allowed.
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["ls".to_string(), "echo".to_string()]);
    let r = bash.execute("ls -la /tmp");
    // Just checking no CommandDenied is returned — the command may or may not succeed
    // depending on the environment, but it must not be blocked.
    assert!(r.await.is_ok(), "ls with args should pass allowlist check");
}

#[tokio::test]
async fn test_allowlist_blocks_unlisted_command() {
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string()]);
    let result = bash.execute("cat /etc/hostname").await;
    assert!(result.is_err(), "non-allowlisted command must be rejected");
    match result.unwrap_err() {
        BashError::CommandDenied(_) => {}
        other => panic!("expected CommandDenied, got: {}", other),
    }
}

#[tokio::test]
async fn test_allowlist_is_case_insensitive() {
    // The check lowercases both sides, so "ECHO" should match "echo".
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string()]);
    let r = bash.execute("ECHO hello").await;
    // Whether ECHO exists on the system is irrelevant; the allowlist must NOT block it.
    // If the system doesn't have ECHO uppercase it will give a 127, not a CommandDenied.
    match r {
        Ok(_) => {}                             // ran, fine
        Err(BashError::CommandDenied(_)) => {
            panic!("case-insensitive allowlist should not block ECHO when echo is listed")
        }
        Err(_) => {} // other errors (e.g. Timeout, IO) are acceptable
    }
}

#[tokio::test]
async fn test_allowlist_multiple_entries_all_work() {
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string(), "printf".to_string(), "true".to_string()]);
    assert!(bash.execute("echo hi").await.is_ok());
    assert!(bash.execute("printf hi").await.is_ok());
    assert!(bash.execute("true").await.is_ok());
}

#[tokio::test]
async fn test_empty_allowlist_blocks_everything() {
    // An empty Vec means "allow nothing".
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec![]);
    let r = bash.execute("echo hello").await;
    assert!(r.is_err(), "empty allowlist should block all commands");
    match r.unwrap_err() {
        BashError::CommandDenied(_) => {}
        other => panic!("expected CommandDenied, got: {}", other),
    }
}

// ─── 4. Denylist security ────────────────────────────────────────────────────

#[tokio::test]
async fn test_denylist_blocks_exact_command() {
    let bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec!["rm".to_string()]);
    let r = bash.execute("rm -rf /").await;
    assert!(r.is_err());
    match r.unwrap_err() {
        BashError::CommandDenied(_) => {}
        other => panic!("expected CommandDenied, got: {}", other),
    }
}

#[tokio::test]
async fn test_denylist_prefix_match_with_args() {
    // "rm" denied means "rm -rf /tmp/x" must also be blocked.
    let bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec!["rm".to_string()]);
    assert!(bash.execute("rm -rf /tmp/x").await.is_err());
    assert!(bash.execute("rm somefile").await.is_err());
}

#[tokio::test]
async fn test_denylist_permits_unlisted_command() {
    let bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec!["rm".to_string()]);
    let r = bash.execute("echo safe").await;
    assert!(r.is_ok());
    assert!(r.unwrap().success);
}

#[tokio::test]
async fn test_denylist_is_case_insensitive() {
    let bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec!["rm".to_string()]);
    // "RM -rf /" should also be blocked.
    let r = bash.execute("RM -rf /").await;
    match r {
        Err(BashError::CommandDenied(_)) => {}  // correct
        Ok(_) => panic!("RM should be blocked by case-insensitive denylist"),
        Err(other) => panic!("unexpected error: {}", other),
    }
}

#[tokio::test]
async fn test_denylist_multiple_entries() {
    let bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec![
            "rm".to_string(),
            "sudo".to_string(),
            "mkfs".to_string(),
        ]);
    assert!(bash.execute("rm file").await.is_err());
    assert!(bash.execute("sudo su").await.is_err());
    assert!(bash.execute("mkfs.ext4 /dev/sda").await.is_err());
    // safe command still works
    assert!(bash.execute("echo hello").await.is_ok());
}

#[tokio::test]
async fn test_denylist_wins_over_allowlist() {
    // "rm" is in BOTH lists — denylist must take priority.
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string(), "rm".to_string()])
        .with_denied_commands(vec!["rm".to_string()]);
    assert!(
        bash.execute("rm file").await.is_err(),
        "denylist must override allowlist for the same prefix"
    );
    // But echo (only in allowlist, not denied) should still work.
    assert!(bash.execute("echo ok").await.is_ok());
}

#[tokio::test]
async fn test_leading_whitespace_does_not_bypass_denylist() {
    // is_command_allowed trims the input, so "  rm -rf /" must still be blocked.
    let bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec!["rm".to_string()]);
    let r = bash.execute("   rm -rf /").await;
    assert!(r.is_err(), "leading whitespace must not bypass the denylist");
}

/// Security note — documented known limitation:
///
/// `is_command_allowed` checks only the *prefix* of the command string.
/// A command like `"echo hello && rm -rf /"` starts with `"echo"` and
/// therefore passes an allowlist that only contains `"echo"`.  Callers
/// should treat the allowlist as a first-line guard, not a complete
/// sandbox.  Full sandboxing requires OS-level mechanisms (seccomp,
/// pledge, containers, etc.).
#[tokio::test]
async fn test_allowlist_prefix_only_does_not_stop_chained_commands() {
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string()]);
    // This command starts with "echo" so it passes the allowlist check,
    // and then the shell executes the chained rm (which will fail harmlessly
    // because the target path doesn't exist, but will NOT be blocked by cloudllm).
    let r = bash
        .execute("echo hello && rm /nonexistent_path_xyz_123")
        .await;
    // The allowlist check passes; rm runs and fails at the OS level (exit 1 or similar).
    // The important thing is that CommandDenied was NOT returned.
    match r {
        Err(BashError::CommandDenied(_)) => {
            panic!(
                "prefix-only allowlist should NOT block chained commands — \
                 this test documents the known limitation"
            )
        }
        _ => {} // OK or OS-level error is the expected behaviour for now
    }
}

// ─── 5. Environment variables ─────────────────────────────────────────────────

#[tokio::test]
async fn test_single_env_var_is_visible() {
    let bash = BashTool::new(Platform::Linux)
        .with_env_var("MY_VAR".to_string(), "my_value".to_string());
    let r = bash.execute("echo $MY_VAR").await.unwrap();
    assert!(r.success);
    assert_eq!(r.stdout.trim(), "my_value");
}

#[tokio::test]
async fn test_multiple_env_vars_are_all_visible() {
    let bash = BashTool::new(Platform::Linux)
        .with_env_var("VAR_A".to_string(), "alpha".to_string())
        .with_env_var("VAR_B".to_string(), "beta".to_string())
        .with_env_var("VAR_C".to_string(), "gamma".to_string());
    let r = bash.execute("echo $VAR_A $VAR_B $VAR_C").await.unwrap();
    assert!(r.success);
    let out = r.stdout.trim();
    assert!(out.contains("alpha"));
    assert!(out.contains("beta"));
    assert!(out.contains("gamma"));
}

#[tokio::test]
async fn test_env_var_with_spaces_in_value() {
    let bash = BashTool::new(Platform::Linux)
        .with_env_var("GREETING".to_string(), "hello world".to_string());
    let r = bash.execute("echo \"$GREETING\"").await.unwrap();
    assert!(r.success);
    assert!(r.stdout.contains("hello world"));
}

// ─── 6. Platform ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_platform_linux() {
    let bash = BashTool::new(Platform::Linux);
    assert_eq!(bash.platform(), Platform::Linux);
    assert_eq!(bash.platform().shell_path(), "/bin/bash");
}

#[tokio::test]
async fn test_platform_macos() {
    let bash = BashTool::new(Platform::macOS);
    assert_eq!(bash.platform(), Platform::macOS);
    assert_eq!(bash.platform().shell_path(), "/bin/bash");
}

#[tokio::test]
async fn test_default_platform_is_linux() {
    let bash = BashTool::default();
    assert_eq!(bash.platform(), Platform::Linux);
    assert_eq!(bash.timeout_secs(), 30);
}

#[tokio::test]
async fn test_platform_actually_runs_bash() {
    // Confirm the tool really executes through bash (not sh or zsh).
    let bash = BashTool::new(if is_macos() { Platform::macOS } else { Platform::Linux });
    let r = bash.execute("echo $BASH_VERSION").await.unwrap();
    assert!(r.success);
    // BASH_VERSION is non-empty only when running under bash.
    assert!(
        !r.stdout.trim().is_empty(),
        "BASH_VERSION should be set when running through bash"
    );
}

// ─── 7. CWD restriction ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_cwd_restriction_sets_working_directory() {
    // The shell's $PWD must equal the configured restriction.
    let bash = BashTool::new(Platform::Linux)
        .with_cwd_restriction(PathBuf::from("/tmp"));
    let r = bash.execute("pwd").await.unwrap();
    assert!(r.success);
    // On macOS /tmp is a symlink to /private/tmp; canonicalise both sides.
    let got = r.stdout.trim().to_string();
    let got_canon = std::fs::canonicalize(&got).unwrap_or_else(|_| PathBuf::from(&got));
    let exp_canon =
        std::fs::canonicalize("/tmp").unwrap_or_else(|_| PathBuf::from("/tmp"));
    assert_eq!(
        got_canon, exp_canon,
        "cwd restriction must set the process working directory"
    );
}

#[tokio::test]
async fn test_cwd_restriction_files_are_relative_to_restricted_dir() {
    // Creating a file with a relative path must land inside the restricted dir.
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_path_buf();

    let bash = BashTool::new(Platform::Linux)
        .with_cwd_restriction(dir_path.clone());
    let r = bash.execute("touch cwd_test_marker.txt && echo ok").await.unwrap();
    assert!(r.success);
    assert!(
        dir_path.join("cwd_test_marker.txt").exists(),
        "file created with relative path must land in the restricted dir"
    );
}

// ─── 8. Thread safety — concurrent execution ─────────────────────────────────

#[tokio::test]
async fn test_shared_tool_survives_concurrent_execution() {
    let bash = Arc::new(BashTool::new(Platform::Linux).with_timeout(10));
    let mut handles = Vec::new();

    for i in 0u32..10 {
        let b = bash.clone();
        handles.push(tokio::spawn(async move {
            let r = b.execute(&format!("echo {}", i)).await.unwrap();
            assert!(r.success);
            assert_eq!(r.stdout.trim(), i.to_string());
        }));
    }

    for h in handles {
        h.await.expect("task should not panic");
    }
}

#[tokio::test]
async fn test_tool_is_reusable_across_many_sequential_calls() {
    let bash = BashTool::new(Platform::Linux);
    for i in 0..20 {
        let r = bash.execute(&format!("echo {}", i)).await.unwrap();
        assert!(r.success);
        assert_eq!(r.stdout.trim(), i.to_string());
    }
}

// ─── 9. BashProtocol (ToolProtocol integration) ──────────────────────────────

#[tokio::test]
async fn test_protocol_name_is_bash() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let proto = BashProtocol::new(bash_tool);
    assert_eq!(proto.protocol_name(), "bash");
}

#[tokio::test]
async fn test_list_tools_returns_one_bash_tool() {
    let proto = BashProtocol::new(Arc::new(BashTool::new(Platform::Linux)));
    let tools = proto.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "bash");
    assert!(!tools[0].description.is_empty());
    assert_eq!(tools[0].parameters.len(), 1);
    assert_eq!(tools[0].parameters[0].name, "command");
    assert!(tools[0].parameters[0].required);
}

#[tokio::test]
async fn test_get_tool_metadata_bash() {
    let proto = BashProtocol::new(Arc::new(BashTool::new(Platform::Linux)));
    let meta = proto.get_tool_metadata("bash").await.unwrap();
    assert_eq!(meta.name, "bash");
    assert!(meta.parameters.iter().any(|p| p.name == "command" && p.required));
}

#[tokio::test]
async fn test_get_tool_metadata_unknown_returns_err() {
    let proto = BashProtocol::new(Arc::new(BashTool::new(Platform::Linux)));
    let r = proto.get_tool_metadata("nonexistent_tool").await;
    assert!(r.is_err(), "unknown tool name must return Err");
}

#[tokio::test]
async fn test_protocol_execute_success_returns_tool_result_success() {
    let proto = BashProtocol::new(Arc::new(BashTool::new(Platform::Linux)));
    let tr = proto
        .execute("bash", serde_json::json!({"command": "echo hi"}))
        .await
        .unwrap();
    assert!(tr.success);
    assert_eq!(tr.output["stdout"].as_str().unwrap().trim(), "hi");
    assert_eq!(tr.output["exit_code"].as_i64().unwrap(), 0);
    assert!(tr.output["duration_ms"].as_u64().unwrap() > 0);
}

/// Bash commands that exit with non-zero are "completed" at the protocol level:
/// `ToolResult.success` is `true` (the *call* succeeded) but `exit_code` is non-zero.
#[tokio::test]
async fn test_protocol_execute_nonzero_exit_returns_tool_success() {
    let proto = BashProtocol::new(Arc::new(BashTool::new(Platform::Linux)));
    // `false` exits 1 — the protocol call itself does not error.
    let tr = proto
        .execute("bash", serde_json::json!({"command": "false"}))
        .await
        .unwrap();
    assert!(
        tr.success,
        "ToolResult.success reflects whether the protocol call succeeded, \
         not whether the command exited 0"
    );
    assert_ne!(
        tr.output["exit_code"].as_i64().unwrap(),
        0,
        "exit_code in the output payload must reflect the command's actual exit code"
    );
}

/// Denied commands translate to `Ok(ToolResult::failure)`, NOT a protocol-level `Err`.
/// Callers should inspect `tr.success` and `tr.error`, not `is_err()`.
#[tokio::test]
async fn test_protocol_execute_denied_command_returns_tool_failure() {
    let proto = BashProtocol::new(Arc::new(
        BashTool::new(Platform::Linux)
            .with_denied_commands(vec!["rm".to_string()]),
    ));
    let tr = proto
        .execute("bash", serde_json::json!({"command": "rm -rf /"}))
        .await
        .unwrap(); // method must return Ok(...)
    assert!(!tr.success, "denied command must yield ToolResult failure");
    assert!(
        tr.error.as_deref().unwrap_or("").contains("denied")
            || tr.error.as_deref().unwrap_or("").contains("Command"),
        "error message should mention denial; got: {:?}",
        tr.error
    );
}

/// Timed-out commands also translate to `Ok(ToolResult::failure)`.
#[tokio::test]
async fn test_protocol_execute_timeout_returns_tool_failure() {
    let proto = BashProtocol::new(Arc::new(
        BashTool::new(Platform::Linux).with_timeout(1),
    ));
    let tr = proto
        .execute("bash", serde_json::json!({"command": "sleep 10"}))
        .await
        .unwrap();
    assert!(!tr.success, "timed-out command must yield ToolResult failure");
    assert!(
        tr.error.as_deref().unwrap_or("").to_lowercase().contains("timeout"),
        "error message should mention timeout; got: {:?}",
        tr.error
    );
}

#[tokio::test]
async fn test_protocol_execute_missing_command_param_returns_err() {
    let proto = BashProtocol::new(Arc::new(BashTool::new(Platform::Linux)));
    // No "command" key in the parameters object.
    let r = proto
        .execute("bash", serde_json::json!({"not_command": "echo hi"}))
        .await;
    assert!(r.is_err(), "missing 'command' parameter must return Err");
}

#[tokio::test]
async fn test_protocol_execute_stderr_captured_in_output() {
    let proto = BashProtocol::new(Arc::new(BashTool::new(Platform::Linux)));
    let tr = proto
        .execute(
            "bash",
            serde_json::json!({"command": "echo err_text >&2"}),
        )
        .await
        .unwrap();
    assert!(tr.success);
    assert!(
        tr.output["stderr"]
            .as_str()
            .unwrap_or("")
            .contains("err_text"),
        "stderr must be surfaced in ToolResult output"
    );
}
