//! Bash Tool Basic Example
//!
//! This example demonstrates how to use the BashTool for basic command execution
//! on Linux and macOS systems.

use cloudllm::tools::{BashTool, Platform};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cloudllm::init_logger();

    println!("=== Bash Tool Basic Example ===\n");

    // Create a bash tool for Linux
    let bash = BashTool::new(Platform::Linux);

    println!("1. Simple echo command:");
    let result = bash.execute("echo 'Hello from bash!'").await?;
    println!("   Output: {}", result.stdout.trim());
    println!("   Success: {}", result.success);
    println!("   Duration: {}ms\n", result.duration_ms);

    // Command with environment variable
    println!("2. Command with environment variable:");
    let bash = BashTool::new(Platform::Linux)
        .with_env_var("GREETING".to_string(), "Welcome".to_string());
    let result = bash.execute("echo $GREETING to Bash Tool").await?;
    println!("   Output: {}", result.stdout.trim());
    println!();

    // Multiple commands
    println!("3. Complex command:");
    let result = bash.execute("for i in {1..3}; do echo \"Item $i\"; done").await?;
    println!("   Output:\n{}", result.stdout);

    // With error handling
    println!("4. Command that fails:");
    let result = bash.execute("false").await?;
    println!("   Success: {}", result.success);
    println!("   Exit code: {}\n", result.exit_code);

    // Stdout and stderr
    println!("5. Capturing stdout and stderr:");
    let result = bash.execute("echo stdout_line && echo stderr_line >&2").await?;
    println!("   Stdout: {}", result.stdout.trim());
    println!("   Stderr: {}", result.stderr.trim());
    println!();

    // With timeout
    println!("6. With custom timeout:");
    let bash_with_timeout = BashTool::new(Platform::Linux).with_timeout(30);
    let result = bash_with_timeout.execute("echo 'Quick command'").await?;
    println!("   Output: {}", result.stdout.trim());
    println!("   Completed within 30s timeout\n");

    // Denied commands
    println!("7. Security - Denied commands:");
    let restricted_bash = BashTool::new(Platform::Linux)
        .with_denied_commands(vec!["rm".to_string(), "sudo".to_string()]);

    let safe_result = restricted_bash.execute("echo 'This is safe'").await;
    println!("   Safe command executed: {}", safe_result.is_ok());

    let dangerous_result = restricted_bash.execute("rm -rf /").await;
    println!("   Dangerous command blocked: {}", dangerous_result.is_err());
    println!();

    // Allowed commands
    println!("8. Security - Allowed commands:");
    let whitelist_bash = BashTool::new(Platform::Linux)
        .with_allowed_commands(vec!["echo".to_string(), "ls".to_string()]);

    let allowed = whitelist_bash.execute("echo 'allowed'").await;
    println!("   Allowed command: {}", allowed.is_ok());

    let not_allowed = whitelist_bash.execute("grep pattern file").await;
    println!("   Not-allowed command blocked: {}", not_allowed.is_err());

    println!("\n=== Summary ===");
    println!("✓ Basic command execution");
    println!("✓ Environment variable support");
    println!("✓ Complex bash commands");
    println!("✓ Error handling");
    println!("✓ Stdout/stderr capture");
    println!("✓ Timeout configuration");
    println!("✓ Security features (deny/allow lists)");

    Ok(())
}
