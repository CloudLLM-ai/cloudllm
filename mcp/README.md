# cloudllm_mcp

`cloudllm_mcp` is the published package name for the standalone Rust crate
that exposes the `mcp` library target. Code can still import it as `mcp`.

`mcp` is a standalone Rust crate containing reusable MCP-oriented tool
protocol primitives, an HTTP client/server runtime, and supporting utilities.

It exists so multiple projects in this repository can share the same MCP
foundation without creating circular dependencies.

## What It Provides

- tool protocol primitives:
  - `ToolProtocol`
  - `ToolMetadata`
  - `ToolParameter`
  - `ToolResult`
  - `ToolRegistry`
- MCP events:
  - `McpEvent`
  - `McpEventHandler`
- MCP runtime:
  - `UnifiedMcpServer`
  - `MCPServerBuilder`
  - `McpClientProtocol`
- server utilities:
  - `HttpServerAdapter`
  - `HttpServerConfig`
  - `HttpServerInstance`
  - `IpFilter`
  - `AuthConfig`

## Build

```bash
cargo build -p cloudllm_mcp
```

Build with HTTP server support:

```bash
cargo build -p cloudllm_mcp --features server
```

## Test

```bash
cargo test -p cloudllm_mcp
```

## Intended Use

- `cloudllm` depends on `mcp` for its shared MCP protocol/runtime layer
- `thoughtchain` depends on `mcp` for its MCP-facing tool schema and result types

This crate is transport/runtime infrastructure, not a complete agent framework.
