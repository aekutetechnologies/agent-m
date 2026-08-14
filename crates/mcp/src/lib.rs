//! Minimal MCP (Model Context Protocol) client for agent-m.
//!
//! Implements just enough of the spec to be a *client* of the common
//! transports — stdio (LSP-style Content-Length framing) and Streamable HTTP
//! (JSON-RPC over POST with SSE responses):
//!
//! - `initialize` handshake with a client-info block
//! - `notifications/initialized`
//! - `tools/list` → MCP tool definitions
//! - `tools/call` → text results / errors
//!
//! Everything else (prompts, resources, sampling) is deliberately out of
//! scope for now — agent-m exposes MCP *servers'* tools to the model.

pub mod client;
pub mod jsonrpc;
pub mod registry;
pub mod tools;

pub use client::McpClient;
pub use registry::{
    McpServerConfig, McpServers, ensure_default_mcp, load_servers, matches_patterns,
};
pub use tools::{McpTool, connect_tools, mcp_tool_spec};
