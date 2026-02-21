//! File System Tool
//!
//! This module provides a safe, restricted file system tool for agents to read, write, and
//! manage files within designated paths. It prevents directory traversal attacks and enforces
//! security restrictions.
//!
//! # Features
//!
//! - **Safe path handling**: Prevents directory traversal attacks (`../../../etc/passwd`)
//! - **Path restriction**: Optional root path to restrict all operations
//! - **File operations**: Read, write, append, delete files
//! - **Directory operations**: List, create, delete directories
//! - **Metadata access**: File size, modification time, permissions, is_directory
//! - **File search**: Find files matching patterns
//! - **Extension filtering**: Optional file extension whitelist
//! - **Error handling**: Comprehensive error types with context
//!
//! # Security
//!
//! - All paths are normalized and validated
//! - Paths that escape the root directory are rejected
//! - No execution of file contents
//! - Optional extension filtering to prevent execution of dangerous files
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```ignore
//! use cloudllm::tools::FileSystemTool;
//! use std::path::PathBuf;
//!
//! let fs = FileSystemTool::new()
//!     .with_root_path(PathBuf::from("/home/user/documents"));
//!
//! // Read a file
//! let content = fs.read_file("notes.txt").await?;
//! println!("Content: {}", content);
//!
//! // Write a file
//! fs.write_file("output.txt", "Hello, World!").await?;
//!
//! // List directory
//! let entries = fs.read_directory(".", false).await?;
//! for entry in entries {
//!     println!("{}: {} bytes", entry.name, entry.size);
//! }
//! ```
//!
//! ## With Agent Integration
//!
//! ```ignore
//! use cloudllm::Agent;
//! use cloudllm::tools::FileSystemTool;
//! use cloudllm::tool_protocols::CustomToolProtocol;
//! use std::path::PathBuf;
//! use std::sync::Arc;
//!
//! let fs_tool = Arc::new(FileSystemTool::new()
//!     .with_root_path(PathBuf::from("/var/data")));
//!
//! // Register with agent's tool registry...
//! # let client = todo!();
//! let agent = Agent::new("analyst", "Data Analyst", client);
//!     // .with_tools(...);
//! ```

use chrono::{DateTime, Local};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// Errors that can occur during file system operations
#[derive(Debug, Clone)]
pub enum FileSystemError {
    /// Path escapes the allowed root directory (security violation)
    PathTraversal(String),
    /// Path does not exist
    NotFound(String),
    /// File is a directory, but a file operation was attempted
    IsDirectory(String),
    /// Path is a directory, but a file operation was attempted
    NotADirectory(String),
    /// File already exists when it shouldn't
    AlreadyExists(String),
    /// Permission denied
    PermissionDenied(String),
    /// File extension not allowed
    ExtensionNotAllowed(String),
    /// IO error with context
    IOError(String),
    /// Invalid path format
    InvalidPath(String),
}

impl fmt::Display for FileSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileSystemError::PathTraversal(msg) => {
                write!(f, "Path traversal attempt blocked: {}", msg)
            }
            FileSystemError::NotFound(msg) => write!(f, "File not found: {}", msg),
            FileSystemError::IsDirectory(msg) => write!(f, "Is a directory: {}", msg),
            FileSystemError::NotADirectory(msg) => write!(f, "Not a directory: {}", msg),
            FileSystemError::AlreadyExists(msg) => write!(f, "Already exists: {}", msg),
            FileSystemError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            FileSystemError::ExtensionNotAllowed(msg) => {
                write!(f, "Extension not allowed: {}", msg)
            }
            FileSystemError::IOError(msg) => write!(f, "IO error: {}", msg),
            FileSystemError::InvalidPath(msg) => write!(f, "Invalid path: {}", msg),
        }
    }
}

impl Error for FileSystemError {}

/// Metadata about a file or directory
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// File or directory name
    pub name: String,
    /// Full path relative to root
    pub path: String,
    /// Size in bytes
    pub size: u64,
    /// Whether this is a directory
    pub is_directory: bool,
    /// Last modified time
    pub modified: String,
}

/// Entry in a directory listing
#[derive(Debug, Clone)]
pub struct DirectoryEntry {
    /// Entry name
    pub name: String,
    /// Whether this is a directory
    pub is_directory: bool,
    /// Size in bytes (0 for directories)
    pub size: u64,
}

/// Safe file system tool for agents with path restrictions
#[derive(Clone)]
pub struct FileSystemTool {
    /// Root path restricting all operations
    root_path: Option<PathBuf>,
    /// Allowed file extensions (None = all allowed)
    allowed_extensions: Option<Vec<String>>,
}

impl FileSystemTool {
    /// Create a new file system tool with no restrictions
    pub fn new() -> Self {
        Self {
            root_path: None,
            allowed_extensions: None,
        }
    }

    /// Set the root path - all operations are restricted to this directory and its subdirectories
    pub fn with_root_path(mut self, path: PathBuf) -> Self {
        self.root_path = Some(path);
        self
    }

    /// Set allowed file extensions (e.g., ["txt", "pdf", "md"])
    pub fn with_allowed_extensions(mut self, extensions: Vec<String>) -> Self {
        self.allowed_extensions = Some(extensions);
        self
    }

    /// Normalize and validate a path
    fn validate_path(&self, path: &str) -> Result<PathBuf, FileSystemError> {
        // Convert to PathBuf
        let path_buf = PathBuf::from(path);

        // Reject absolute paths
        if path_buf.is_absolute() {
            return Err(FileSystemError::InvalidPath(
                "Absolute paths are not allowed".to_string(),
            ));
        }

        // Resolve .. and . components relative to root
        let mut normalized = PathBuf::new();
        for component in path_buf.components() {
            use std::path::Component;
            match component {
                Component::ParentDir => {
                    normalized.pop();
                }
                Component::Normal(c) => normalized.push(c),
                Component::CurDir => {} // Skip . components
                _ => {} // Ignore other components (shouldn't happen for relative paths)
            }
        }

        // Get the effective path (with root if set)
        let effective_path = if let Some(root) = &self.root_path {
            root.join(&normalized)
        } else {
            normalized
        };

        // Verify the effective path is within root (if root is set).
        //
        // Always canonicalize to resolve symlinks before comparing against root_canonical.
        // For paths that don't exist yet (write/create), canonicalize the nearest existing
        // ancestor and reconstruct the non-existent suffix under it — this prevents symlink
        // escapes through parent directory components.
        if let Some(root) = &self.root_path {
            let root_canonical = root.canonicalize().map_err(|e| {
                FileSystemError::IOError(format!("Cannot canonicalize root: {}", e))
            })?;

            let canonical_to_check = if effective_path.exists() {
                // Path exists — canonicalize fully (resolves all symlinks).
                effective_path.canonicalize().map_err(|e| {
                    FileSystemError::IOError(format!("Cannot canonicalize path: {}", e))
                })?
            } else {
                // Path doesn't exist yet (write/create).
                // Canonicalize the nearest existing ancestor to catch symlinks in parent dirs.
                let parent = effective_path.parent().ok_or_else(|| {
                    FileSystemError::InvalidPath("Path has no parent".to_string())
                })?;
                let canonical_parent = if parent.exists() {
                    parent.canonicalize().map_err(|e| {
                        FileSystemError::IOError(format!("Cannot canonicalize parent: {}", e))
                    })?
                } else {
                    // Walk up until we find an existing ancestor.
                    let mut ancestor = parent;
                    loop {
                        if ancestor.exists() {
                            break ancestor.canonicalize().map_err(|e| {
                                FileSystemError::IOError(format!(
                                    "Cannot canonicalize ancestor: {}",
                                    e
                                ))
                            })?;
                        }
                        ancestor = ancestor.parent().ok_or_else(|| {
                            FileSystemError::InvalidPath(
                                "No existing ancestor found".to_string(),
                            )
                        })?;
                    }
                };
                // Reconstruct the non-existent suffix under the canonical parent.
                let suffix = effective_path.strip_prefix(parent).unwrap_or(&effective_path);
                canonical_parent.join(suffix)
            };

            if !canonical_to_check.starts_with(&root_canonical) {
                return Err(FileSystemError::PathTraversal(format!(
                    "Path escapes root directory: {}",
                    path
                )));
            }
        }

        Ok(effective_path)
    }

    /// Check if file extension is allowed
    fn check_extension(&self, path: &Path) -> Result<(), FileSystemError> {
        if let Some(allowed) = &self.allowed_extensions {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if !allowed.iter().any(|a| a.to_lowercase() == ext_str) {
                    return Err(FileSystemError::ExtensionNotAllowed(format!(
                        "Extension .{} not allowed",
                        ext_str
                    )));
                }
            }
        }
        Ok(())
    }

    /// Read entire file content as string
    pub async fn read_file(&self, path: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;
        self.check_extension(&safe_path)?;

        if !safe_path.exists() {
            return Err(Box::new(FileSystemError::NotFound(path.to_string())));
        }

        if safe_path.is_dir() {
            return Err(Box::new(FileSystemError::IsDirectory(path.to_string())));
        }

        let content = fs::read_to_string(&safe_path).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        Ok(content)
    }

    /// Write content to file (overwrites if exists)
    pub async fn write_file(
        &self,
        path: &str,
        content: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;
        self.check_extension(&safe_path)?;

        if safe_path.exists() && safe_path.is_dir() {
            return Err(Box::new(FileSystemError::IsDirectory(path.to_string())));
        }

        // Ensure parent directory exists
        if let Some(parent) = safe_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
            })?;
        }

        fs::write(&safe_path, content).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        Ok(())
    }

    /// Append content to file
    pub async fn append_file(
        &self,
        path: &str,
        content: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;
        self.check_extension(&safe_path)?;

        if safe_path.exists() && safe_path.is_dir() {
            return Err(Box::new(FileSystemError::IsDirectory(path.to_string())));
        }

        // Ensure parent directory exists
        if let Some(parent) = safe_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
            })?;
        }

        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&safe_path)
            .map_err(|e| {
                Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
            })?;

        file.write_all(content.as_bytes()).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        Ok(())
    }

    /// Get file metadata
    pub async fn get_file_metadata(
        &self,
        path: &str,
    ) -> Result<FileMetadata, Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;

        if !safe_path.exists() {
            return Err(Box::new(FileSystemError::NotFound(path.to_string())));
        }

        let metadata = fs::metadata(&safe_path).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        let modified_time = metadata
            .modified()
            .ok()
            .and_then(|t| {
                DateTime::<Local>::from(t)
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string()
                    .parse()
                    .ok()
            })
            .unwrap_or_else(|| "unknown".to_string());

        Ok(FileMetadata {
            name: safe_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            path: path.to_string(),
            size: metadata.len(),
            is_directory: metadata.is_dir(),
            modified: modified_time,
        })
    }

    /// Read directory contents
    pub async fn read_directory(
        &self,
        path: &str,
        recursive: bool,
    ) -> Result<Vec<DirectoryEntry>, Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;

        if !safe_path.exists() {
            return Err(Box::new(FileSystemError::NotFound(path.to_string())));
        }

        if !safe_path.is_dir() {
            return Err(Box::new(FileSystemError::NotADirectory(path.to_string())));
        }

        let mut entries = Vec::new();

        if recursive {
            self.read_directory_recursive(&safe_path, &mut entries)?;
        } else {
            for entry in fs::read_dir(&safe_path).map_err(|e| {
                Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
            })? {
                let entry = entry.map_err(|e| {
                    Box::new(FileSystemError::IOError(e.to_string()))
                        as Box<dyn Error + Send + Sync>
                })?;
                let metadata = entry.metadata().map_err(|e| {
                    Box::new(FileSystemError::IOError(e.to_string()))
                        as Box<dyn Error + Send + Sync>
                })?;

                entries.push(DirectoryEntry {
                    name: entry.file_name().to_string_lossy().to_string(),
                    is_directory: metadata.is_dir(),
                    size: if metadata.is_dir() { 0 } else { metadata.len() },
                });
            }
        }

        Ok(entries)
    }

    /// Recursively read directory
    #[allow(clippy::only_used_in_recursion)]
    fn read_directory_recursive(
        &self,
        path: &Path,
        entries: &mut Vec<DirectoryEntry>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        for entry in fs::read_dir(path).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })? {
            let entry = entry.map_err(|e| {
                Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
            })?;
            let metadata = entry.metadata().map_err(|e| {
                Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
            })?;

            entries.push(DirectoryEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                is_directory: metadata.is_dir(),
                size: if metadata.is_dir() { 0 } else { metadata.len() },
            });

            if metadata.is_dir() {
                // Validate the resolved path before recursing to prevent a symlink inside the
                // tree from pointing outside the root and being silently traversed.
                if let Some(root) = &self.root_path {
                    if let Ok(root_canonical) = root.canonicalize() {
                        match entry.path().canonicalize() {
                            Ok(canonical) if !canonical.starts_with(&root_canonical) => {
                                // Symlink points outside the root — skip silently.
                                continue;
                            }
                            Err(_) => continue, // Cannot resolve — skip.
                            Ok(_) => {}          // Within root — proceed.
                        }
                    }
                }
                self.read_directory_recursive(&entry.path(), entries)?;
            }
        }

        Ok(())
    }

    /// Delete a file
    pub async fn delete_file(&self, path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;

        if !safe_path.exists() {
            return Err(Box::new(FileSystemError::NotFound(path.to_string())));
        }

        if safe_path.is_dir() {
            return Err(Box::new(FileSystemError::IsDirectory(path.to_string())));
        }

        fs::remove_file(&safe_path).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        Ok(())
    }

    /// Delete a directory (recursively)
    pub async fn delete_directory(&self, path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;

        if !safe_path.exists() {
            return Err(Box::new(FileSystemError::NotFound(path.to_string())));
        }

        if !safe_path.is_dir() {
            return Err(Box::new(FileSystemError::NotADirectory(path.to_string())));
        }

        fs::remove_dir_all(&safe_path).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        Ok(())
    }

    /// Check if a file or directory exists
    pub async fn file_exists(&self, path: &str) -> Result<bool, Box<dyn Error + Send + Sync>> {
        match self.validate_path(path) {
            Ok(safe_path) => Ok(safe_path.exists()),
            Err(_) => Ok(false), // Path traversal or invalid = doesn't exist
        }
    }

    /// Create a directory (with parents)
    pub async fn create_directory(&self, path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;

        if safe_path.exists() {
            if !safe_path.is_dir() {
                return Err(Box::new(FileSystemError::AlreadyExists(path.to_string())));
            }
            return Ok(()); // Already exists and is a directory
        }

        fs::create_dir_all(&safe_path).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        Ok(())
    }

    /// Search for files matching a name pattern (simple substring matching)
    pub async fn search_files(
        &self,
        directory: &str,
        pattern: &str,
    ) -> Result<Vec<DirectoryEntry>, Box<dyn Error + Send + Sync>> {
        let entries = self.read_directory(directory, true).await?;
        let matching: Vec<_> = entries
            .into_iter()
            .filter(|e| e.name.contains(pattern))
            .collect();

        Ok(matching)
    }

    /// Get file size in bytes
    pub async fn get_file_size(&self, path: &str) -> Result<u64, Box<dyn Error + Send + Sync>> {
        let safe_path = self.validate_path(path)?;

        if !safe_path.exists() {
            return Err(Box::new(FileSystemError::NotFound(path.to_string())));
        }

        if safe_path.is_dir() {
            return Err(Box::new(FileSystemError::IsDirectory(path.to_string())));
        }

        let metadata = fs::metadata(&safe_path).map_err(|e| {
            Box::new(FileSystemError::IOError(e.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        Ok(metadata.len())
    }
}

impl Default for FileSystemTool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_filesystem_creation() {
        let fs = FileSystemTool::new();
        assert!(fs.root_path.is_none());
        assert!(fs.allowed_extensions.is_none());
    }

    #[tokio::test]
    async fn test_with_root_path() {
        let path = PathBuf::from("/tmp");
        let fs = FileSystemTool::new().with_root_path(path.clone());
        assert_eq!(fs.root_path, Some(path));
    }

    #[tokio::test]
    async fn test_with_allowed_extensions() {
        let fs = FileSystemTool::new()
            .with_allowed_extensions(vec!["txt".to_string(), "md".to_string()]);
        assert!(fs.allowed_extensions.is_some());
    }

    #[tokio::test]
    async fn test_write_and_read_file() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        // Write file
        fs.write_file("test.txt", "Hello, World!").await.unwrap();

        // Read file
        let content = fs.read_file("test.txt").await.unwrap();
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_append_file() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("test.txt", "Hello").await.unwrap();
        fs.append_file("test.txt", " World").await.unwrap();

        let content = fs.read_file("test.txt").await.unwrap();
        assert_eq!(content, "Hello World");
    }

    #[tokio::test]
    async fn test_file_exists() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("test.txt", "content").await.unwrap();

        assert!(fs.file_exists("test.txt").await.unwrap());
        assert!(!fs.file_exists("nonexistent.txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_delete_file() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("test.txt", "content").await.unwrap();
        assert!(fs.file_exists("test.txt").await.unwrap());

        fs.delete_file("test.txt").await.unwrap();
        assert!(!fs.file_exists("test.txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_get_file_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("test.txt", "Hello").await.unwrap();
        let metadata = fs.get_file_metadata("test.txt").await.unwrap();

        assert_eq!(metadata.name, "test.txt");
        assert_eq!(metadata.size, 5);
        assert!(!metadata.is_directory);
    }

    #[tokio::test]
    async fn test_create_directory() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.create_directory("subdir").await.unwrap();
        assert!(fs.file_exists("subdir").await.unwrap());
    }

    #[tokio::test]
    async fn test_read_directory() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("file1.txt", "content1").await.unwrap();
        fs.write_file("file2.txt", "content2").await.unwrap();
        fs.create_directory("subdir").await.unwrap();

        let entries = fs.read_directory(".", false).await.unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn test_get_file_size() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("test.txt", "Hello").await.unwrap();
        let size = fs.get_file_size("test.txt").await.unwrap();
        assert_eq!(size, 5);
    }

    #[tokio::test]
    async fn test_path_traversal_prevention() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        // Try to escape the root directory
        let result = fs.read_file("../../../etc/passwd").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extension_filtering() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new()
            .with_root_path(temp_dir.path().to_path_buf())
            .with_allowed_extensions(vec!["txt".to_string()]);

        fs.write_file("test.txt", "content").await.unwrap();
        let result = fs.write_file("test.pdf", "content");
        assert!(result.await.is_err());
    }

    #[tokio::test]
    async fn test_delete_directory() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.create_directory("subdir").await.unwrap();
        fs.write_file("subdir/file.txt", "content").await.unwrap();

        fs.delete_directory("subdir").await.unwrap();
        assert!(!fs.file_exists("subdir").await.unwrap());
    }

    #[tokio::test]
    async fn test_search_files() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("test1.txt", "content").await.unwrap();
        fs.write_file("test2.txt", "content").await.unwrap();
        fs.write_file("other.md", "content").await.unwrap();

        let results = fs.search_files(".", "test").await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        let result = fs.read_file("nonexistent.txt").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_directory_recursive() {
        let temp_dir = TempDir::new().unwrap();
        let fs = FileSystemTool::new().with_root_path(temp_dir.path().to_path_buf());

        fs.write_file("file1.txt", "content").await.unwrap();
        fs.create_directory("subdir").await.unwrap();
        fs.write_file("subdir/file2.txt", "content").await.unwrap();

        let entries = fs.read_directory(".", true).await.unwrap();
        assert!(entries.len() >= 3);
    }
}
