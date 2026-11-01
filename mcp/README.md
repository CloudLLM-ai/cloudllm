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
- streamable HTTP transport:
  - `StreamableHttpConfig`
  - `streamable_http_router`
  - `streamable_http_router_with_sse`
- SSE (Server-Sent Events) support:
  - `SseBroadcaster` — multi-subscriber broadcast channel for SSE events
  - `SseEventHandler` — bridges `McpEvent` lifecycle events into the SSE stream
  - `SseMessage` — typed SSE message with optional event name and JSON data
  - GET endpoint returns `text/event-stream` when a broadcaster is attached
  - POST requests automatically broadcast JSON-RPC results/errors as SSE events

## Browser-facing protections

Both transports (legacy `/tools/*` + `/resources/*` routes and streamable HTTP)
apply the same checks, in order: IP filter, `Host`, authentication, `Origin`.

- **Origin**: a present `Origin` header must be a loopback origin
  (`localhost`, `127.0.0.1`, `[::1]`, any port). Opt out with
  `with_skip_origin_validation(true)` for LAN browser clients. Requests with no
  `Origin` (non-browser clients) are unaffected.
- **Host** (DNS-rebinding defence): bound to a loopback address, only
  `localhost`, `127.0.0.1` and `[::1]` are accepted. Behind a reverse proxy,
  list your public names with `MCPServerBuilder::with_allowed_hosts`,
  `StreamableHttpConfig::with_allowed_hosts` or
  `RequestGuard::with_allowed_hosts`; the list replaces the default. A
  non-loopback bind without a list performs no Host check.
- **No authentication is an explicit choice**: with no bearer token and no
  authorizer every client that passes the checks above may run tools. The
  Axum adapter prints a startup warning in that case. Configure a token for
  anything that is not a throwaway local server.
- 5xx responses carry a generic message plus a request id; the real error goes
  to stderr with the same id.

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
- `mentisdb` depends on `mcp` for its MCP-facing tool schema and result types

This crate is transport/runtime infrastructure, not a complete agent framework.
