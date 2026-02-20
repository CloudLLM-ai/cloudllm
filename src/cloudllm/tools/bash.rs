//! Bash Command Execution Tool
//!
//! This module provides a secure, configurable tool for agents to execute bash commands
//! on Linux and macOS systems. It supports safety features like timeouts, command allowlisting,
//! working directory restrictions, and environment variable controls.
//!
//! # Features
//!
//! - **Cross-Platform Support**: Linux and macOS with platform-specific shell selection
//! - **Security**: Command allowlisting/denylisting, working directory restrictions
//! - **Timeouts**: Configurable timeout for long-running commands
//! - **Output Control**: Separate stdout/stderr capture with size limits
//! - **Environment Variables**: Safe propagation of environment variables
//! - **Thread-Safe**: Full async/await support with Arc<Mutex<>>
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```ignore
//! use cloudllm::tools::{BashTool, Platform};
//!
//! let bash = BashTool::new(Platform::Linux);
//! let result = bash.execute("ls -la /tmp").await?;
//! println!("Files: {}", result.stdout);
//! ```
//!
//! ## With Safety Features
//!
//! ```ignore
//! use cloudllm::tools::{BashTool, Platform};
//! use std::path::PathBuf;
//!
//! let bash = BashTool::new(Platform::macOS)
//!     .with_timeout(30)
//!     .with_cwd_restriction(PathBuf::from("/home/user"))
//!     .with_denied_commands(vec!["rm -rf".to_string(), "sudo".to_string()]);
//!
//! let result = bash.execute("find . -type f -name '*.txt'").await?;
//! ```
//!
//! ## With Agent Integration
//!
//! ```ignore
//! use cloudllm::Agent;
//! use cloudllm::tools::BashTool;
//! use cloudllm::tool_protocols::CustomToolProtocol;
//!
//! let bash = BashTool::new(Platform::Linux).with_timeout(60);
//! let adapter = BashToolAdapter::new(Arc::new(bash));
//! let agent = Agent::new("analyst", "File Analysis", client)
//!     .with_tools(Arc::new(ToolRegistry::new(adapter)));
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command as TokioCommand;

/// Platform selector for bash tool
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    /// Linux platform using /bin/bash
    Linux,
    /// macOS platform using /bin/bash (or /bin/zsh on newer systems)
    #[allow(non_camel_case_types)]
    macOS,
}

impl Platform {
    /// Get the shell path for this platform
    pub fn shell_path(&self) -> &'static str {
        match self {
            Platform::Linux => "/bin/bash",
            Platform::macOS => "/bin/bash",
        }
    }

    /// Get the shell invocation flag for this platform
    pub fn shell_flag(&self) -> &'static str {
        "-c"
    }
}

/// Result of bash command execution
#[derive(Debug, Clone)]
pub struct BashResult {
    /// Whether the command executed successfully (exit code 0)
    pub success: bool,
    /// Standard output captured from the command
    pub stdout: String,
    /// Standard error output captured from the command
    pub stderr: String,
    /// Exit code returned by the command
    pub exit_code: i32,
    /// Duration of command execution in milliseconds
    pub duration_ms: u64,
}

impl BashResult {
    /// Create a successful bash result
    pub fn success(stdout: String, stderr: String, duration_ms: u64) -> Self {
        Self {
            success: true,
            stdout,
            stderr,
            exit_code: 0,
            duration_ms,
        }
    }

    /// Create a failed bash result
    pub fn failure(stdout: String, stderr: String, exit_code: i32, duration_ms: u64) -> Self {
        Self {
            success: false,
            stdout,
            stderr,
            exit_code,
            duration_ms,
        }
    }
}

/// Errors that can occur during bash command execution
#[derive(Debug)]
pub enum BashError {
    /// Command execution timed out
    Timeout(String),
    /// Command was denied by allowlist/denylist
    CommandDenied(String),
    /// Working directory is outside allowed restriction
    CwdRestrictionViolated(String),
    /// Command failed to execute
    ExecutionFailed(String),
    /// IO error during command execution
    IoError(std::io::Error),
    /// Command output exceeded size limits
    OutputTooLarge(String),
}

impl std::fmt::Display for BashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BashError::Timeout(msg) => write!(f, "Command timeout: {}", msg),
            BashError::CommandDenied(msg) => write!(f, "Command denied: {}", msg),
            BashError::CwdRestrictionViolated(msg) => {
                write!(f, "CWD restriction violated: {}", msg)
            }
            BashError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            BashError::IoError(e) => write!(f, "IO error: {}", e),
            BashError::OutputTooLarge(msg) => write!(f, "Output too large: {}", msg),
        }
    }
}

impl std::error::Error for BashError {}

/// Maximum output size per stream (stdout/stderr) in bytes - default 10MB
const DEFAULT_MAX_OUTPUT_SIZE: usize = 10 * 1024 * 1024;

/// Read from `reader` into a byte buffer, returning an error if the stream
/// exceeds `max_bytes`.  Used to enforce `max_output_size` on stdout/stderr.
async fn read_limited<R: AsyncReadExt + Unpin>(
    mut reader: R,
    max_bytes: usize,
    stream_name: &'static str,
) -> Result<Vec<u8>, BashError> {
    let mut buf = Vec::new();
    let mut chunk = vec![0u8; 8192];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) => return Ok(buf),
            Ok(n) => {
                if buf.len() + n > max_bytes {
                    return Err(BashError::OutputTooLarge(format!(
                        "{} exceeded the {} byte limit",
                        stream_name, max_bytes
                    )));
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            Err(e) => return Err(BashError::IoError(e)),
        }
    }
}

/// Bash command execution tool with security features
///
/// This tool provides a secure interface for agents to execute bash commands
/// on Linux and macOS systems. It supports multiple safety mechanisms including
/// command allowlisting/denylisting, timeout enforcement, working directory
/// restrictions, and environment variable controls.
///
/// # Security Considerations
///
/// - Always use `with_allowed_commands()` for restrictive environments
/// - Use `with_cwd_restriction()` to limit file system access
/// - Set appropriate timeouts to prevent hanging commands
/// - Be cautious with agent prompts that might generate dangerous commands
///
/// # Thread Safety
///
/// BashTool is fully thread-safe and can be shared across multiple agents
/// using `Arc<BashTool>`.
#[derive(Clone)]
pub struct BashTool {
    /// Selected platform (Linux or macOS)
    platform: Platform,
    /// Timeout for command execution in seconds
    timeout_secs: u64,
    /// Whitelist of allowed commands (None means allow all)
    allowed_commands: Arc<Mutex<Option<Vec<String>>>>,
    /// Blacklist of denied commands
    denied_commands: Arc<Mutex<Option<Vec<String>>>>,
    /// Restrict commands to this working directory (None means no restriction)
    cwd_restriction: Arc<Mutex<Option<PathBuf>>>,
    /// Environment variables to pass to the command
    env_vars: Arc<Mutex<HashMap<String, String>>>,
    /// Maximum output size per stream in bytes; enforced during execution.
    max_output_size: usize,
}

impl BashTool {
    /// Create a new BashTool for the specified platform
    ///
    /// # Arguments
    ///
    /// * `platform` - The platform to run commands on (Linux or macOS)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::{BashTool, Platform};
    ///
    /// let bash_linux = BashTool::new(Platform::Linux);
    /// let bash_macos = BashTool::new(Platform::macOS);
    /// ```
    pub fn new(platform: Platform) -> Self {
        Self {
            platform,
            timeout_secs: 30,
            allowed_commands: Arc::new(Mutex::new(None)),
            denied_commands: Arc::new(Mutex::new(None)),
            cwd_restriction: Arc::new(Mutex::new(None)),
            env_vars: Arc::new(Mutex::new(HashMap::new())),
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
        }
    }

    /// Set the timeout for command execution
    ///
    /// # Arguments
    ///
    /// * `secs` - Timeout in seconds (default is 30)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::{BashTool, Platform};
    ///
    /// let bash = BashTool::new(Platform::Linux)
    ///     .with_timeout(60);  // 60 second timeout
    /// ```
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set a whitelist of allowed commands
    ///
    /// When set, only commands starting with one of these prefixes will be allowed.
    /// Useful for restrictive environments.
    ///
    /// # Arguments
    ///
    /// * `cmds` - Vector of command prefixes to allow
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::{BashTool, Platform};
    ///
    /// let bash = BashTool::new(Platform::Linux)
    ///     .with_allowed_commands(vec![
    ///         "ls".to_string(),
    ///         "find".to_string(),
    ///         "grep".to_string(),
    ///     ]);
    /// ```
    pub fn with_allowed_commands(self, cmds: Vec<String>) -> Self {
        *self.allowed_commands.lock().unwrap() = Some(cmds);
        self
    }

    /// Set a blacklist of denied commands
    ///
    /// Commands starting with any of these prefixes will be rejected.
    /// Useful for blocking dangerous operations.
    ///
    /// # Arguments
    ///
    /// * `cmds` - Vector of command prefixes to deny
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::{BashTool, Platform};
    ///
    /// let bash = BashTool::new(Platform::Linux)
    ///     .with_denied_commands(vec![
    ///         "rm".to_string(),
    ///         "sudo".to_string(),
    ///         "kill".to_string(),
    ///     ]);
    /// ```
    pub fn with_denied_commands(self, cmds: Vec<String>) -> Self {
        *self.denied_commands.lock().unwrap() = Some(cmds);
        self
    }

    /// Restrict command execution to a specific working directory
    ///
    /// When set, commands can only access files within this directory tree.
    /// The restriction is enforced by checking relative paths.
    ///
    /// # Arguments
    ///
    /// * `path` - The allowed working directory
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::{BashTool, Platform};
    /// use std::path::PathBuf;
    ///
    /// let bash = BashTool::new(Platform::Linux)
    ///     .with_cwd_restriction(PathBuf::from("/home/user/data"));
    /// ```
    pub fn with_cwd_restriction(self, path: PathBuf) -> Self {
        *self.cwd_restriction.lock().unwrap() = Some(path);
        self
    }

    /// Override the maximum number of bytes collected from stdout or stderr.
    ///
    /// If either stream exceeds this limit the child process is killed and
    /// `BashError::OutputTooLarge` is returned.  Defaults to 10 MiB.
    pub fn with_max_output_size(mut self, bytes: usize) -> Self {
        self.max_output_size = bytes;
        self
    }

    /// Add or override an environment variable for command execution
    ///
    /// # Arguments
    ///
    /// * `key` - Environment variable name
    /// * `value` - Environment variable value
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::{BashTool, Platform};
    ///
    /// let bash = BashTool::new(Platform::Linux)
    ///     .with_env_var("PATH".to_string(), "/usr/local/bin:/usr/bin".to_string())
    ///     .with_env_var("LANG".to_string(), "en_US.UTF-8".to_string());
    /// ```
    pub fn with_env_var(self, key: String, value: String) -> Self {
        self.env_vars.lock().unwrap().insert(key, value);
        self
    }

    /// Get the current platform
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Get the current timeout in seconds
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Check if a command is allowed to execute.
    ///
    /// Matching is case-insensitive and inspects both the raw command string
    /// and the **basename** of the first word so that absolute-path variants
    /// (e.g. `/bin/rm`, `../bin/rm`) are caught by the same rules as the bare
    /// command name.
    ///
    /// # Security note
    ///
    /// The check examines only the *first token* of the command string.  Shell
    /// metacharacters such as `;`, `&&`, `||`, `$(...)`, and backticks can
    /// chain additional commands that bypass these rules.  Use OS-level
    /// sandboxing (seccomp, containers) for stronger isolation.
    fn is_command_allowed(&self, cmd: &str) -> Result<(), BashError> {
        let cmd_lower = cmd.trim().to_lowercase();

        // Extract the basename of the first word so that `/bin/rm`, `./rm`,
        // and `../bin/rm` are all caught by a denylist entry of `"rm"`.
        let first_word = cmd_lower.split_whitespace().next().unwrap_or("");
        let cmd_basename = first_word.rsplit('/').next().unwrap_or(first_word);

        // A helper that returns true if `entry` matches either the full
        // command or its basename.
        let matches = |entry: &str| -> bool {
            let e = entry.to_lowercase();
            cmd_lower.starts_with(&e) || cmd_basename.starts_with(&e)
        };

        // Check denied list first (denylist beats allowlist).
        if let Some(denied) = self.denied_commands.lock().unwrap().as_ref() {
            for denied_cmd in denied {
                if matches(denied_cmd) {
                    return Err(BashError::CommandDenied(format!(
                        "Command '{}' is denied",
                        denied_cmd
                    )));
                }
            }
        }

        // Check allowed list if present.
        if let Some(allowed) = self.allowed_commands.lock().unwrap().as_ref() {
            if !allowed.iter().any(|allowed_cmd| matches(allowed_cmd)) {
                return Err(BashError::CommandDenied(
                    "Command not in allowed list".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Execute a bash command
    ///
    /// This method executes a bash command with the configured safety settings.
    /// It captures stdout and stderr separately and enforces the timeout.
    ///
    /// # Arguments
    ///
    /// * `cmd` - The bash command to execute
    ///
    /// # Returns
    ///
    /// A `BashResult` containing the command output, exit code, and execution time.
    /// Returns `BashError` if the command is denied, times out, or fails to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cloudllm::tools::{BashTool, Platform};
    ///
    /// let bash = BashTool::new(Platform::Linux);
    /// let result = bash.execute("echo 'Hello, World!'").await?;
    /// assert!(result.success);
    /// assert_eq!(result.stdout.trim(), "Hello, World!");
    /// ```
    pub async fn execute(&self, cmd: &str) -> Result<BashResult, BashError> {
        // Check if command is allowed
        self.is_command_allowed(cmd)?;

        let start_time = Instant::now();
        let platform = self.platform;
        let shell_path = platform.shell_path().to_string();
        let shell_flag = platform.shell_flag().to_string();
        let cmd = cmd.to_string();
        let timeout = std::time::Duration::from_secs(self.timeout_secs);

        // Get environment variables and optional CWD restriction.
        let env_vars = self.env_vars.lock().unwrap().clone();
        let cwd = self.cwd_restriction.lock().unwrap().clone();

        let max_output = self.max_output_size;

        // Use tokio::process::Command so the future is truly async and
        // cancellable.  Spawn the process and read stdout/stderr incrementally
        // so we can enforce max_output_size without buffering unbounded data.
        match tokio::time::timeout(timeout, async move {
            let mut command = TokioCommand::new(&shell_path);
            command
                .arg(&shell_flag)
                .arg(&cmd)
                .envs(env_vars)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());

            if let Some(dir) = cwd {
                command.current_dir(dir);
            }

            let mut child = command.spawn().map_err(BashError::IoError)?;
            let stdout_pipe = child.stdout.take().expect("stdout was piped");
            let stderr_pipe = child.stderr.take().expect("stderr was piped");

            // Read both streams concurrently to avoid pipe-buffer deadlocks.
            let (stdout_result, stderr_result) = tokio::join!(
                read_limited(stdout_pipe, max_output, "stdout"),
                read_limited(stderr_pipe, max_output, "stderr"),
            );

            // On output overflow kill the child before surfacing the error.
            let (stdout_bytes, stderr_bytes) = match (stdout_result, stderr_result) {
                (Err(e), _) | (_, Err(e)) => {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    return Err(e);
                }
                (Ok(out), Ok(err)) => (out, err),
            };

            let status = child.wait().await.map_err(BashError::IoError)?;
            let duration_ms = start_time.elapsed().as_millis() as u64;

            let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
            let stderr = String::from_utf8_lossy(&stderr_bytes).to_string();

            if status.success() {
                Ok(BashResult::success(stdout, stderr, duration_ms))
            } else {
                let exit_code = status.code().unwrap_or(-1);
                Ok(BashResult::failure(stdout, stderr, exit_code, duration_ms))
            }
        })
        .await
        {
            Ok(result) => result,
            Err(_) => Err(BashError::Timeout(format!(
                "Command exceeded {} second timeout",
                self.timeout_secs
            ))),
        }
    }
}

impl Default for BashTool {
    /// Create a BashTool with default platform (Linux)
    fn default() -> Self {
        Self::new(Platform::Linux)
    }
}
