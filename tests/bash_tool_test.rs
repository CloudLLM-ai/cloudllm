use cloudllm::tools::{BashTool, Platform};

#[tokio::test]
async fn test_bash_tool_simple_echo() {
    let bash = BashTool::new(Platform::Linux);
    let result = bash.execute("echo hello").await.unwrap();

    assert!(result.success);
    assert_eq!(result.stdout.trim(), "hello");
    assert_eq!(result.exit_code, 0);
}

#[tokio::test]
async fn test_bash_tool_successful_command() {
    let bash = BashTool::new(Platform::Linux);
    let result = bash.execute("echo 'test output'").await.unwrap();

    assert!(result.success);
    assert!(result.stdout.contains("test output"));
    assert_eq!(result.exit_code, 0);
    assert!(result.duration_ms > 0);
}

#[tokio::test]
async fn test_bash_tool_failed_command() {
    let bash = BashTool::new(Platform::Linux);
    let result = bash.execute("false").await.unwrap();

    assert!(!result.success);
    assert_ne!(result.exit_code, 0);
}

#[tokio::test]
async fn test_bash_tool_with_custom_timeout() {
    let bash = BashTool::new(Platform::Linux).with_timeout(10);
    assert_eq!(bash.timeout_secs(), 10);

    let result = bash.execute("echo quick").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_bash_tool_timeout_configuration() {
    // Test that we can configure different timeouts
    let bash_10s = BashTool::new(Platform::Linux).with_timeout(10);
    let bash_60s = BashTool::new(Platform::Linux).with_timeout(60);

    assert_eq!(bash_10s.timeout_secs(), 10);
    assert_eq!(bash_60s.timeout_secs(), 60);

    // Both should successfully execute quick commands
    let result1 = bash_10s.execute("echo test").await.unwrap();
    let result2 = bash_60s.execute("echo test").await.unwrap();

    assert!(result1.success);
    assert!(result2.success);
}

#[tokio::test]
async fn test_bash_tool_allowed_commands() {
    let bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string(), "ls".to_string()]);

    let result = bash.execute("echo allowed").await;
    assert!(result.is_ok());

    let result = bash.execute("rm -rf /").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_bash_tool_denied_commands() {
    let bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec!["rm".to_string(), "sudo".to_string()]);

    let result = bash.execute("echo safe").await;
    assert!(result.is_ok());

    let result = bash.execute("rm test.txt").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_bash_tool_platform_linux() {
    let bash = BashTool::new(Platform::Linux);
    assert_eq!(bash.platform(), Platform::Linux);
    assert_eq!(bash.platform().shell_path(), "/bin/bash");
}

#[tokio::test]
async fn test_bash_tool_platform_macos() {
    let bash = BashTool::new(Platform::macOS);
    assert_eq!(bash.platform(), Platform::macOS);
    assert_eq!(bash.platform().shell_path(), "/bin/bash");
}

#[tokio::test]
async fn test_bash_tool_with_environment_variables() {
    let bash = BashTool::new(Platform::Linux)
        .with_env_var("TEST_VAR".to_string(), "test_value".to_string());

    let result = bash.execute("echo $TEST_VAR").await.unwrap();
    assert!(result.success);
    assert!(result.stdout.contains("test_value"));
}

#[tokio::test]
async fn test_bash_tool_stderr_capture() {
    let bash = BashTool::new(Platform::Linux);
    let result = bash.execute("echo error >&2").await.unwrap();

    assert!(result.success);
    assert!(result.stderr.contains("error"));
}

#[tokio::test]
async fn test_bash_tool_stdout_and_stderr() {
    let bash = BashTool::new(Platform::Linux);
    let result = bash.execute("echo out; echo err >&2").await.unwrap();

    assert!(result.success);
    assert!(result.stdout.contains("out"));
    assert!(result.stderr.contains("err"));
}

#[tokio::test]
async fn test_bash_tool_default_platform() {
    let bash = BashTool::default();
    assert_eq!(bash.platform(), Platform::Linux);
}

#[tokio::test]
async fn test_bash_tool_multiple_commands() {
    let bash = BashTool::new(Platform::Linux);

    let result1 = bash.execute("echo first").await.unwrap();
    assert!(result1.success);

    let result2 = bash.execute("echo second").await.unwrap();
    assert!(result2.success);

    let result3 = bash.execute("echo third").await.unwrap();
    assert!(result3.success);
}

#[tokio::test]
async fn test_bash_tool_complex_command() {
    let bash = BashTool::new(Platform::Linux);
    let result = bash
        .execute("for i in 1 2 3; do echo $i; done")
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.stdout.contains("1"));
    assert!(result.stdout.contains("2"));
    assert!(result.stdout.contains("3"));
}
