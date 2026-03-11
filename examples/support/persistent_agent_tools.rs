#![allow(dead_code)]

use cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult,
};
use cloudllm::tool_protocols::{
    BashProtocol, CustomToolProtocol, HttpClientProtocol, McpClientProtocol, MemoryProtocol,
};
use cloudllm::tools::{BashTool, Calculator, FileSystemTool, HttpClient, Memory, Platform};
use serde_json::{json, Value};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

pub async fn build_persistent_agent_registry(
    thoughtchain_endpoint: &str,
    filesystem_root: PathBuf,
) -> Result<(ToolRegistry, Arc<McpClientProtocol>), Box<dyn Error + Send + Sync>> {
    let thoughtchain_protocol =
        Arc::new(McpClientProtocol::new(thoughtchain_endpoint.to_string()).with_cache_ttl(30));

    let mut registry = ToolRegistry::empty();
    registry
        .add_protocol("thoughtchain", thoughtchain_protocol.clone())
        .await?;

    let memory = Arc::new(Memory::new());
    registry
        .add_protocol("memory", Arc::new(MemoryProtocol::new(memory)))
        .await?;

    let bash_tool = Arc::new(BashTool::new(detect_platform()).with_timeout(30));
    registry
        .add_protocol("bash", Arc::new(BashProtocol::new(bash_tool)))
        .await?;

    let http_client = Arc::new(HttpClient::new());
    registry
        .add_protocol("http", Arc::new(HttpClientProtocol::new(http_client)))
        .await?;

    let custom = Arc::new(CustomToolProtocol::new());
    register_calculator_tool(custom.clone()).await;
    register_filesystem_tools(custom.clone(), filesystem_root).await;
    registry.add_protocol("custom", custom).await?;

    Ok((registry, thoughtchain_protocol))
}

async fn register_calculator_tool(protocol: Arc<CustomToolProtocol>) {
    let calculator = Arc::new(Calculator::new());
    protocol
        .register_async_tool(
            ToolMetadata::new("calculator", "Evaluate a mathematical expression.").with_parameter(
                ToolParameter::new("expression", ToolParameterType::String)
                    .with_description("Expression to evaluate, e.g. 'sqrt(16) + mean([1,2,3])'.")
                    .required(),
            ),
            Arc::new(move |params| {
                let calculator = calculator.clone();
                Box::pin(async move {
                    let expression = required_string(&params, "expression")?;
                    match calculator.evaluate(&expression).await {
                        Ok(result) => Ok(ToolResult::success(json!({ "result": result }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;
}

async fn register_filesystem_tools(protocol: Arc<CustomToolProtocol>, filesystem_root: PathBuf) {
    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));

    protocol
        .register_async_tool(
            ToolMetadata::new(
                "read_file",
                "Read a text file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.read_file(&path).await {
                        Ok(content) => Ok(ToolResult::success(json!({ "content": content }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "write_file",
                "Write a text file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String)
                    .with_description("Text content to write.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    let content = required_string(&params, "content")?;
                    match filesystem.write_file(&path, &content).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "append_file",
                "Append text to a file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String)
                    .with_description("Text content to append.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    let content = required_string(&params, "content")?;
                    match filesystem.append_file(&path, &content).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "list_directory",
                "List a directory inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the directory.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("recursive", ToolParameterType::Boolean)
                    .with_description("Whether to recurse into subdirectories."),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    let recursive = params
                        .get("recursive")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    match filesystem.read_directory(&path, recursive).await {
                        Ok(entries) => Ok(ToolResult::success(json!({
                            "entries": entries
                                .iter()
                                .map(directory_entry_to_json)
                                .collect::<Vec<_>>()
                        }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "file_metadata",
                "Return file metadata inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.get_file_metadata(&path).await {
                        Ok(metadata) => Ok(ToolResult::success(json!({
                            "metadata": file_metadata_to_json(&metadata)
                        }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "create_directory",
                "Create a directory inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the directory.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.create_directory(&path).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "delete_file",
                "Delete a file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.delete_file(&path).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "delete_directory",
                "Delete a directory inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the directory.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.delete_directory(&path).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "search_files",
                "Search for files by substring within the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("directory", ToolParameterType::String)
                    .with_description("Relative path to the directory to search.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("pattern", ToolParameterType::String)
                    .with_description("Substring to search for in file names.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let directory = required_string(&params, "directory")?;
                    let pattern = required_string(&params, "pattern")?;
                    match filesystem.search_files(&directory, &pattern).await {
                        Ok(entries) => Ok(ToolResult::success(json!({
                            "entries": entries
                                .iter()
                                .map(directory_entry_to_json)
                                .collect::<Vec<_>>()
                        }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "file_exists",
                "Check whether a path exists inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to check.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.file_exists(&path).await {
                        Ok(exists) => Ok(ToolResult::success(json!({ "exists": exists }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "file_size",
                "Return a file size in bytes inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.get_file_size(&path).await {
                        Ok(size) => Ok(ToolResult::success(json!({ "size": size }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;
}

fn required_string(parameters: &Value, key: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    parameters
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("'{key}' parameter is required").into())
}

fn detect_platform() -> Platform {
    #[cfg(target_os = "macos")]
    {
        Platform::macOS
    }
    #[cfg(not(target_os = "macos"))]
    {
        Platform::Linux
    }
}

fn directory_entry_to_json(entry: &cloudllm::tools::DirectoryEntry) -> Value {
    json!({
        "name": entry.name,
        "is_directory": entry.is_directory,
        "size": entry.size,
    })
}

fn file_metadata_to_json(metadata: &cloudllm::tools::FileMetadata) -> Value {
    json!({
        "name": metadata.name,
        "path": metadata.path,
        "size": metadata.size,
        "is_directory": metadata.is_directory,
        "modified": metadata.modified,
    })
}
