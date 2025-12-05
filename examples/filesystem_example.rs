//! File System Tool Example
//!
//! This example demonstrates the FileSystemTool capabilities:
//! - Safe file operations (read, write, append, delete)
//! - Directory management (create, list, delete)
//! - Metadata retrieval and file search
//! - Path traversal prevention
//! - Extension filtering for security

use cloudllm::tools::FileSystemTool;
use std::path::Path;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== CloudLLM File System Tool Example ===\n");

    // Create a temporary directory for demo (in production, use a real path)
    let temp_dir = TempDir::new()?;
    let root_path = temp_dir.path().to_path_buf();
    println!("Working directory: {:?}\n", root_path);

    // Example 1: Basic file operations
    basic_file_operations(&root_path).await?;

    // Example 2: Directory management
    directory_management(&root_path).await?;

    // Example 3: File search and metadata
    file_search_and_metadata(&root_path).await?;

    // Example 4: Extension filtering for security
    extension_filtering(&root_path).await?;

    // Example 5: Error handling
    error_handling(&root_path).await?;

    // Example 6: Bulk operations
    bulk_operations(&root_path).await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

async fn basic_file_operations(
    root_path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("--- Example 1: Basic File Operations ---");

    let fs = FileSystemTool::new().with_root_path(root_path.to_path_buf());

    // Write a file
    fs.write_file("config.txt", "environment=production\nversion=1.0")
        .await?;
    println!("✓ Created config.txt");

    // Read the file
    let content = fs.read_file("config.txt").await?;
    println!("✓ Read config.txt: {}", content.lines().next().unwrap());

    // Append to file
    fs.append_file("config.txt", "\nstatus=active").await?;
    println!("✓ Appended to config.txt");

    // Check if file exists
    if fs.file_exists("config.txt").await? {
        println!("✓ Confirmed config.txt exists");
    }

    // Get file size
    let size = fs.get_file_size("config.txt").await?;
    println!("✓ File size: {} bytes\n", size);

    Ok(())
}

async fn directory_management(
    root_path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("--- Example 2: Directory Management ---");

    let fs = FileSystemTool::new().with_root_path(root_path.to_path_buf());

    // Create directories
    fs.create_directory("data/logs").await?;
    println!("✓ Created nested directory: data/logs");

    fs.create_directory("data/cache").await?;
    println!("✓ Created directory: data/cache");

    // Write files in subdirectories
    fs.write_file("data/logs/app.log", "[INFO] Application started")
        .await?;
    fs.write_file("data/cache/session_001.cache", "cached_data")
        .await?;
    println!("✓ Created files in subdirectories");

    // List directory contents (non-recursive)
    let entries = fs.read_directory("data", false).await?;
    println!("✓ Contents of 'data' directory: {} entries", entries.len());
    for entry in entries {
        println!("  - {} (dir: {})", entry.name, entry.is_directory);
    }

    // List directory contents (recursive)
    let all_entries = fs.read_directory("data", true).await?;
    println!(
        "✓ All files under 'data' (recursive): {} entries",
        all_entries.len()
    );

    println!();
    Ok(())
}

async fn file_search_and_metadata(
    root_path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("--- Example 3: File Search and Metadata ---");

    let fs = FileSystemTool::new().with_root_path(root_path.to_path_buf());

    // Create test files
    fs.write_file("report_2024_01.txt", "January report")
        .await?;
    fs.write_file("report_2024_02.txt", "February report")
        .await?;
    fs.write_file("summary.txt", "Annual summary").await?;
    println!("✓ Created test files");

    // Search for files
    let matches = fs.search_files(".", "report").await?;
    println!("✓ Found {} files matching 'report'", matches.len());
    for entry in matches {
        println!("  - {}", entry.name);
    }

    // Get file metadata
    let metadata = fs.get_file_metadata("summary.txt").await?;
    println!("\n✓ Metadata for summary.txt:");
    println!("  - Path: {}", metadata.path);
    println!("  - Size: {} bytes", metadata.size);
    println!("  - Is directory: {}", metadata.is_directory);
    println!("  - Modified: {}", metadata.modified);

    println!();
    Ok(())
}

async fn extension_filtering(
    root_path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("--- Example 4: Extension Filtering ---");

    let fs = FileSystemTool::new()
        .with_root_path(root_path.to_path_buf())
        .with_allowed_extensions(vec![
            "txt".to_string(),
            "md".to_string(),
            "json".to_string(),
        ]);

    // Allowed: txt
    match fs.write_file("allowed.txt", "This is allowed").await {
        Ok(_) => println!("✓ Successfully wrote .txt file"),
        Err(e) => println!("✗ Failed to write .txt: {}", e),
    }

    // Allowed: md
    match fs.write_file("readme.md", "# Documentation").await {
        Ok(_) => println!("✓ Successfully wrote .md file"),
        Err(e) => println!("✗ Failed to write .md: {}", e),
    }

    // Not allowed: exe
    match fs.write_file("malware.exe", "evil code").await {
        Ok(_) => println!("✗ Should have blocked .exe file!"),
        Err(_) => println!("✓ Blocked dangerous .exe file"),
    }

    // Not allowed: sh
    match fs.write_file("script.sh", "#!/bin/bash").await {
        Ok(_) => println!("✗ Should have blocked .sh file!"),
        Err(_) => println!("✓ Blocked shell script .sh file"),
    }

    println!();
    Ok(())
}

async fn error_handling(
    root_path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("--- Example 5: Error Handling ---");

    let fs = FileSystemTool::new().with_root_path(root_path.to_path_buf());

    // Try to read non-existent file
    match fs.read_file("nonexistent.txt").await {
        Ok(_) => println!("✗ Should have failed!"),
        Err(_) => println!("✓ Correctly handled missing file error"),
    }

    // Try to read directory as file
    fs.create_directory("mydir").await?;
    match fs.read_file("mydir").await {
        Ok(_) => println!("✗ Should have failed!"),
        Err(_) => println!("✓ Correctly prevented reading directory as file"),
    }

    // Try to write to existing directory
    match fs.write_file("mydir", "content").await {
        Ok(_) => println!("✗ Should have failed!"),
        Err(_) => println!("✓ Correctly prevented writing to directory path"),
    }

    // Try to escape root directory
    match fs.read_file("../../../etc/passwd").await {
        Ok(_) => println!("✗ Security issue: path escape allowed!"),
        Err(_) => println!("✓ Path traversal correctly prevented"),
    }

    println!();
    Ok(())
}

async fn bulk_operations(
    root_path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("--- Example 6: Bulk Operations ---");

    let fs = FileSystemTool::new().with_root_path(root_path.to_path_buf());

    // Create a project structure
    fs.create_directory("project/src").await?;
    fs.create_directory("project/tests").await?;
    fs.create_directory("project/docs").await?;
    println!("✓ Created project structure");

    // Write multiple files
    fs.write_file("project/Cargo.toml", "[package]\nname = \"myapp\"")
        .await?;
    fs.write_file("project/src/main.rs", "fn main() {}").await?;
    fs.write_file("project/README.md", "# My Project").await?;
    println!("✓ Created project files");

    // Get total file count
    let files = fs.read_directory("project", true).await?;
    let file_count = files.iter().filter(|e| !e.is_directory).count();
    let dir_count = files.iter().filter(|e| e.is_directory).count();
    println!(
        "✓ Project contains {} files and {} directories\n",
        file_count, dir_count
    );

    Ok(())
}
