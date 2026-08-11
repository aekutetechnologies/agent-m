//! Web tools: `web_fetch` (read a URL into readable text) and `web_search`
//! (query a configurable JSON search API — SearXNG-style).
//!
//! Both are read-only (Low risk tier). `web_fetch` mirrors the `read` tool's
//! safety caps: 10 MB response limit and a 10-second timeout. `web_search`
//! needs a search endpoint configured via the `AGENT_M_SEARCH_URL` env var
//! (a GET endpoint returning `[{"title", "url", "snippet"}]`) — there is no
//! built-in search API key, so without configuration it fails with a clear
//! setup message instead of pretending.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use serde_json::{Value, json};

const MAX_BYTES: usize = 10 * 1024 * 1024;
const TIMEOUT_SECS: u64 = 10;
/// How much of the fetched text the model sees (tokens are precious).
const MAX_TEXT: usize = 60_000;

pub struct WebFetchTool;

#[async_trait::async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> String {
        "Fetch a URL and return its content as readable text (HTML is stripped \
         to text; plain text and JSON are returned verbatim). Capped at 10 MB \
         with a 10-second timeout. Read-only."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "Absolute http(s) URL to fetch" }
            },
            "required": ["url"]
        })
    }

    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let url = arguments
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if url.is_empty() {
            return Err(ToolError::failed(self.name(), "missing `url` argument"));
        }
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolOutcome {
                content: "web_fetch: refusing non-http(s) URL (e.g. file://, localhost bypasses)"
                    .into(),
                is_error: true,
            });
        }
        if let Err(reason) = url_allowed(url) {
            return Ok(ToolOutcome {
                content: format!("web_fetch: blocked {url} ({reason})"),
                is_error: true,
            });
        }
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
            .user_agent("agent-m/0.1 (web_fetch)")
            // Re-validate every redirect hop (no automatic fetches past the
            // SSRF guard).
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if url_allowed(attempt.url().as_str()).is_ok() {
                    attempt.follow()
                } else {
                    attempt.stop()
                }
            }))
            .build()
            .map_err(|error| ToolError::failed(self.name(), error.to_string()))?;

        let response = match client.get(url).send().await {
            Ok(response) => response,
            Err(error) => {
                return Ok(ToolOutcome {
                    content: format!("web_fetch: request failed: {error}"),
                    is_error: true,
                });
            }
        };
        if !response.status().is_success() {
            return Ok(ToolOutcome {
                content: format!("web_fetch: HTTP {}", response.status()),
                is_error: true,
            });
        }
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = match response.bytes().await {
            Ok(bytes) => bytes,
            Err(error) => {
                return Ok(ToolOutcome {
                    content: format!("web_fetch: read failed: {error}"),
                    is_error: true,
                });
            }
        };
        if bytes.len() > MAX_BYTES {
            return Ok(ToolOutcome {
                content: format!(
                    "web_fetch: response too large ({} bytes > {MAX_BYTES} cap)",
                    bytes.len()
                ),
                is_error: true,
            });
        }
        let text = String::from_utf8_lossy(&bytes);
        let looks_html = text.trim_start().starts_with('<');
        let readable = if content_type.contains("json") {
            text.trim().to_string()
        } else if content_type.contains("html") || looks_html {
            html_to_text(&text)
        } else {
            text.trim().to_string()
        };
        let truncated = readable.chars().take(MAX_TEXT).collect::<String>();
        let suffix = if readable.chars().count() > MAX_TEXT {
            "\n\n… (truncated)".to_string()
        } else {
            String::new()
        };
        Ok(ToolOutcome {
            content: format!("{truncated}{suffix}"),
            is_error: false,
        })
    }
}

/// SSRF guard: refuse loopback, RFC1918, link-local, and CGNAT targets so a
/// prompt-steered or 302'd fetch can't reach internal services (e.g. the
/// cloud metadata endpoint). Hostnames like `localhost`/`*.internal` are
/// blocked by name; DNS-rebinding protection for arbitrary hostnames is a
/// documented limit.
fn url_allowed(url: &str) -> Result<(), &'static str> {
    let Some((_, rest)) = url.split_once("://") else {
        return Err("not an absolute URL");
    };
    let host_port = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .rsplit('@')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("");
    if host_port.is_empty() {
        return Err("no host");
    }
    let host = host_port.to_lowercase();
    if host == "localhost"
        || host.ends_with(".localhost")
        || host == "metadata.google.internal"
        || host.ends_with(".internal")
    {
        return Err("loopback/internal hostname");
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        if ip.is_loopback() {
            return Err("loopback address");
        }
        let is_private = match ip {
            std::net::IpAddr::V4(v4) => {
                let octets = v4.octets();
                octets[0] == 10
                    || octets[0] == 127
                    || (octets[0] == 172 && (16..=31).contains(&octets[1]))
                    || (octets[0] == 192 && octets[1] == 168)
                    || (octets[0] == 169 && octets[1] == 254)
                    || (octets[0] == 100 && (64..=127).contains(&octets[1]))
            }
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback()
                    || (v6.segments()[0] & 0xfe00 == 0xfc00)
                    || (v6.segments()[0] == 0 && v6.segments()[1] == 0)
            }
        };
        if is_private {
            return Err("private/link-local address");
        }
    }
    Ok(())
}

/// Strip HTML tags/entities into readable text (headings gain blank lines,
/// links keep their text, scripts/styles are dropped).
fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut chars = html.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
            }
            continue;
        }
        match ch {
            '<' => {
                // Peek the tag name to skip script/style blocks. Keep the
                // leading '/' for closing tags.
                let rest: String = chars.clone().take(16).collect();
                let name = rest
                    .split(|c: char| c.is_whitespace() || c == '>')
                    .next()
                    .unwrap_or("")
                    .to_lowercase();
                match name.as_str() {
                    "script" => in_script = true,
                    "/script" => in_script = false,
                    "style" => in_style = true,
                    "/style" => in_style = false,
                    "p" | "div" | "li" | "h1" | "h2" | "h3" | "h4" | "br" | "tr" | "pre" => {
                        out.push('\n')
                    }
                    _ => {}
                }
                in_tag = true;
            }
            '&' => {
                // Match the entity up to the first ';' and consume exactly
                // that many characters (never beyond the entity).
                let buffer: String = chars.clone().take(12).collect();
                let semicolon = buffer.find(';');
                let code = semicolon.map(|index| &buffer[..index]).unwrap_or("");
                let decoded = match code {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "quot" => "\"",
                    "apos" => "'",
                    "nbsp" => " ",
                    _ => "&",
                };
                out.push_str(decoded);
                let consume = semicolon.map_or(1, |index| index + 1);
                for _ in 0..consume {
                    chars.next();
                }
            }
            other => {
                if !(in_script || in_style) {
                    out.push(other);
                }
            }
        }
    }
    // Collapse runs of whitespace, keep single blank lines between blocks.
    let mut lines: Vec<String> = out
        .split('\n')
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect();
    // De-duplicate consecutive empty-ish lines by filtering already.
    lines.truncate(400); // sanity cap on the text the model reads
    lines.join("\n")
}

pub struct WebSearchTool;

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> String {
        "Search the web via the configured JSON search endpoint \
         (AGENT_M_SEARCH_URL, SearXNG-style: GET ?q=… returning \
         [{\"title\",\"url\",\"snippet\"}]). Read-only. Returns the top \
         results with title, url, and snippet."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "max_results": { "type": "integer", "description": "Max results (default 5, max 10)" }
            },
            "required": ["query"]
        })
    }

    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if query.is_empty() {
            return Err(ToolError::failed(self.name(), "missing `query` argument"));
        }
        let max_results = arguments
            .get("max_results")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 10) as usize;

        let Some(endpoint) = std::env::var("AGENT_M_SEARCH_URL")
            .ok()
            .filter(|s| !s.is_empty())
        else {
            return Ok(ToolOutcome {
                content: "web_search: no search endpoint configured. Set \
                         AGENT_M_SEARCH_URL to a SearXNG-style JSON API \
                         (GET ?q=… returning [{\"title\",\"url\",\"snippet\"}]), \
                         e.g. http://localhost:8888/search?format=json"
                    .into(),
                is_error: true,
            });
        };

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
            .user_agent("agent-m/0.1 (web_search)")
            .build()
            .map_err(|error| ToolError::failed(self.name(), error.to_string()))?;

        let response = match client
            .get(endpoint)
            .query(&[("q", query.as_str()), ("format", "json")])
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                return Ok(ToolOutcome {
                    content: format!("web_search: request failed: {error}"),
                    is_error: true,
                });
            }
        };
        let body: Value = match response.json().await {
            Ok(body) => body,
            Err(_) => {
                return Ok(ToolOutcome {
                    content: "web_search: endpoint did not return JSON".into(),
                    is_error: true,
                });
            }
        };
        let results = body
            .as_array()
            .or_else(|| body.get("results").and_then(Value::as_array))
            .cloned()
            .unwrap_or_default();
        if results.is_empty() {
            return Ok(ToolOutcome {
                content: "web_search: no results".into(),
                is_error: false,
            });
        }
        let mut lines = Vec::new();
        for result in results.into_iter().take(max_results) {
            let title = result.get("title").and_then(Value::as_str).unwrap_or("");
            let url = result.get("url").and_then(Value::as_str).unwrap_or("");
            let snippet = result.get("snippet").and_then(Value::as_str).unwrap_or("");
            lines.push(format!("- {title}\n  {url}\n  {snippet}"));
        }
        Ok(ToolOutcome {
            content: lines.join("\n"),
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_text_strips_tags_and_keeps_content() {
        let html = "<html><head><title>T</title><style>.x{}</style></head><body>\
                    <h1>Hello</h1><p>Some <b>bold</b> text &amp; more.</p>\
                    <script>alert('x')</script><p>after</p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("bold"));
        assert!(text.contains("&"));
        assert!(text.contains("after"));
        assert!(!text.contains("<"), "no raw tags: {text}");
        assert!(!text.contains("alert"), "script content dropped");
    }

    #[test]
    fn ssrf_guard_blocks_internal_targets() {
        assert_eq!(
            url_allowed("http://localhost:8080/x").unwrap_err(),
            "loopback/internal hostname"
        );
        assert_eq!(
            url_allowed("http://127.0.0.1/x").unwrap_err(),
            "loopback address"
        );
        assert!(url_allowed("http://10.0.0.5/x").is_err());
        assert!(url_allowed("http://169.254.169.254/latest/meta-data").is_err());
        assert!(url_allowed("http://192.168.1.1/x").is_err());
        assert!(url_allowed("http://metadata.google.internal/").is_err());
        assert!(url_allowed("https://example.com/path").is_ok());
        assert!(url_allowed("https://example.com:443/x").is_ok());
        assert!(
            url_allowed("file:///etc/passwd").is_err(),
            "no scheme-relative"
        );
    }

    #[test]
    fn web_fetch_refuses_non_http_urls() {
        let context = ToolContext::simple(std::path::PathBuf::from("."));
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let outcome = runtime
            .block_on(WebFetchTool.execute(json!({ "url": "file:///etc/passwd" }), &context))
            .unwrap();
        assert!(outcome.is_error);
        assert!(outcome.content.contains("refusing non-http(s)"));
    }

    #[test]
    fn web_search_without_endpoint_explains_setup() {
        let previous = std::env::var("AGENT_M_SEARCH_URL").ok();
        if previous.is_some() {
            // Rust 2024: env mutation is unsafe.
            unsafe { std::env::remove_var("AGENT_M_SEARCH_URL") };
        }
        let context = ToolContext::simple(std::path::PathBuf::from("."));
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let outcome = runtime
            .block_on(WebSearchTool.execute(json!({ "query": "rust mcp" }), &context))
            .unwrap();
        assert!(outcome.is_error);
        assert!(outcome.content.contains("AGENT_M_SEARCH_URL"));
        if let Some(value) = previous {
            unsafe { std::env::set_var("AGENT_M_SEARCH_URL", value) };
        }
    }
}
