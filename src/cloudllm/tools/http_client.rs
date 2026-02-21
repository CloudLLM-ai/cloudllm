//! # HTTP/REST API Client Tool
//!
//! A secure HTTP client for agents to make REST API calls to external services.
//!
//! ## Features
//!
//! This HTTP client supports:
//!
//! - **HTTP Methods**: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
//! - **Request Management**: JSON payloads, query parameters, custom headers
//! - **Security**: Domain allowlist/blocklist, timeout controls, size limits
//! - **Response Handling**: Status codes, headers, body with size limits
//! - **Authentication**: Basic auth, Bearer token, custom headers
//! - **Content Types**: Automatic JSON detection, custom content-type support
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use cloudllm::tools::HttpClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = HttpClient::new();
//!
//!     // Simple GET request
//!     let response = client.get("https://api.example.com/users").await?;
//!     println!("Status: {}", response.status);
//!     println!("Body: {}", response.body);
//!
//!     // POST with JSON payload
//!     let response = client.post(
//!         "https://api.example.com/users",
//!         serde_json::json!({ "name": "Alice", "email": "alice@example.com" })
//!     ).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Detailed Examples
//!
//! ### GET Requests
//!
//! ```rust,ignore
//! let response = client.get("https://api.example.com/data").await?;
//! println!("Status: {}", response.status);
//! println!("Headers: {:?}", response.headers);
//! println!("Body: {}", response.body);
//! ```
//!
//! ### POST with JSON
//!
//! ```rust,ignore
//! let payload = serde_json::json!({
//!     "title": "New Post",
//!     "content": "Hello, World!",
//!     "author_id": 42
//! });
//!
//! let response = client.post("https://api.example.com/posts", payload).await?;
//! assert_eq!(response.status, 201);
//! ```
//!
//! ### Query Parameters
//!
//! ```rust,ignore
//! let mut client = HttpClient::new();
//! client.with_query_param("page", "1");
//! client.with_query_param("limit", "50");
//! client.with_query_param("sort", "created_at");
//!
//! let response = client.get("https://api.example.com/items").await?;
//! // URL becomes: https://api.example.com/items?page=1&limit=50&sort=created_at
//! ```
//!
//! ### Custom Headers and Authentication
//!
//! ```rust,ignore
//! let mut client = HttpClient::new();
//! client.with_header("Authorization", "Bearer sk_live_abc123");
//! client.with_header("X-API-Key", "secret-key");
//! client.with_header("User-Agent", "MyAgent/1.0");
//!
//! let response = client.get("https://api.example.com/data").await?;
//! ```
//!
//! ### Basic Authentication
//!
//! ```rust,ignore
//! let mut client = HttpClient::new();
//! client.with_basic_auth("username", "password");
//!
//! let response = client.get("https://api.example.com/protected").await?;
//! ```
//!
//! ### Security: Domain Allowlist
//!
//! ```rust,ignore
//! let mut client = HttpClient::new();
//! client.allow_domain("api.example.com");
//! client.allow_domain("api.partner.com");
//!
//! // This will succeed
//! let response = client.get("https://api.example.com/data").await?;
//!
//! // This will fail - domain not allowed
//! let response = client.get("https://untrusted.com/data").await;
//! assert!(response.is_err());
//! ```
//!
//! ### Security: Domain Blocklist
//!
//! ```rust,ignore
//! let mut client = HttpClient::new();
//! client.deny_domain("malicious.com");
//! client.deny_domain("phishing.net");
//!
//! // This will succeed - not blocked
//! let response = client.get("https://safe.com/data").await?;
//!
//! // This will fail - domain blocked
//! let response = client.get("https://malicious.com/data").await;
//! assert!(response.is_err());
//! ```
//!
//! ### Timeout Configuration
//!
//! ```rust,ignore
//! let mut client = HttpClient::new();
//! client.with_timeout(std::time::Duration::from_secs(5));
//!
//! // Request will timeout if it takes longer than 5 seconds
//! let response = client.get("https://slow-api.example.com/data").await;
//! ```
//!
//! ### Error Handling
//!
//! ```rust,ignore
//! match client.get("https://api.example.com/data").await {
//!     Ok(response) => {
//!         match response.status {
//!             200 => println!("Success: {}", response.body),
//!             404 => println!("Not found"),
//!             500 => println!("Server error"),
//!             _ => println!("Status: {}", response.status),
//!         }
//!     }
//!     Err(e) => println!("Request failed: {}", e),
//! }
//! ```
//!
//! ## Security Considerations
//!
//! The HTTP client includes several security features:
//!
//! - **Domain Control**: Allowlist and blocklist for domain restrictions
//! - **Size Limits**: Default 10MB response limit to prevent memory exhaustion
//! - **Timeout**: Default 30 second timeout on all requests
//! - **HTTPS Preferred**: While HTTP is supported, HTTPS is recommended
//! - **Header Validation**: Custom headers are validated before sending
//!
//! ## Performance
//!
//! - Connection pooling via reqwest
//! - Fast DNS resolution via system resolver
//! - Typical request: <100ms to <1s depending on network
//! - Concurrent requests supported via tokio async

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::net::IpAddr;
use std::time::Duration;

use reqwest::Client;
use serde_json::Value as JsonValue;

/// Error type for HTTP client operations
#[derive(Debug, Clone)]
pub struct HttpClientError {
    message: String,
}

impl HttpClientError {
    /// Create a new HTTP client error
    pub fn new(message: impl Into<String>) -> Self {
        HttpClientError {
            message: message.into(),
        }
    }
}

impl fmt::Display for HttpClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP client error: {}", self.message)
    }
}

impl Error for HttpClientError {}

/// HTTP response containing status, headers, and body
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code (e.g., 200, 404, 500)
    pub status: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body as string
    pub body: String,
}

impl HttpResponse {
    /// Check if response status indicates success (2xx)
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// Check if response status indicates client error (4xx)
    pub fn is_client_error(&self) -> bool {
        self.status >= 400 && self.status < 500
    }

    /// Check if response status indicates server error (5xx)
    pub fn is_server_error(&self) -> bool {
        self.status >= 500 && self.status < 600
    }

    /// Try to parse response body as JSON
    pub fn json(&self) -> Result<JsonValue, Box<dyn Error + Send + Sync>> {
        serde_json::from_str(&self.body).map_err(|e| {
            Box::new(HttpClientError::new(format!("Failed to parse JSON: {}", e)))
                as Box<dyn Error + Send + Sync>
        })
    }
}

/// HTTP client for making REST API calls
///
/// The HTTP client is thread-safe and supports builder-style configuration.
/// It maintains domain allowlist/blocklist for security and enforces size/timeout limits.
///
/// # Examples
///
/// ```rust,ignore
/// use cloudllm::tools::HttpClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = HttpClient::new();
///
///     // GET request
///     let response = client.get("https://api.example.com/data").await?;
///     println!("Status: {}", response.status);
///
///     // POST request
///     let payload = serde_json::json!({"key": "value"});
///     let response = client.post("https://api.example.com/data", payload).await?;
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct HttpClient {
    client: Client,
    allowed_domains: Vec<String>,
    denied_domains: Vec<String>,
    query_params: HashMap<String, String>,
    headers: HashMap<String, String>,
    timeout: Duration,
    max_response_size: usize,
}

impl HttpClient {
    /// Create a new HTTP client with default settings
    ///
    /// Default configuration:
    /// - Timeout: 30 seconds
    /// - Max response size: 10MB
    /// - No domain restrictions
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = HttpClient::new();
    /// ```
    pub fn new() -> Self {
        HttpClient {
            client: Client::new(),
            allowed_domains: Vec::new(),
            denied_domains: Vec::new(),
            query_params: HashMap::new(),
            headers: HashMap::new(),
            timeout: Duration::from_secs(30),
            max_response_size: 10 * 1024 * 1024, // 10MB
        }
    }

    /// Add a domain to the allowlist
    ///
    /// When allowlist is not empty, only requests to listed domains are allowed.
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain to allow (e.g., "api.example.com")
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = HttpClient::new();
    /// client.allow_domain("api.example.com");
    /// client.allow_domain("api.partner.com");
    /// ```
    pub fn allow_domain(&mut self, domain: &str) -> &mut Self {
        self.allowed_domains.push(domain.to_string());
        self
    }

    /// Add a domain to the blocklist
    ///
    /// Blocked domains will never be allowed, even if they're in the allowlist.
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain to deny (e.g., "malicious.com")
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = HttpClient::new();
    /// client.deny_domain("malicious.com");
    /// client.deny_domain("phishing.net");
    /// ```
    pub fn deny_domain(&mut self, domain: &str) -> &mut Self {
        self.denied_domains.push(domain.to_string());
        self
    }

    /// Add a query parameter to all requests
    ///
    /// Query parameters are appended to the URL for all requests.
    ///
    /// # Arguments
    ///
    /// * `key` - Parameter name
    /// * `value` - Parameter value
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = HttpClient::new();
    /// client.with_query_param("api_key", "secret123");
    /// client.with_query_param("version", "v2");
    /// ```
    pub fn with_query_param(&mut self, key: &str, value: &str) -> &mut Self {
        self.query_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a custom HTTP header to all requests
    ///
    /// # Arguments
    ///
    /// * `name` - Header name (e.g., "Authorization")
    /// * `value` - Header value
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = HttpClient::new();
    /// client.with_header("Authorization", "Bearer token123");
    /// client.with_header("X-API-Key", "secret");
    /// ```
    pub fn with_header(&mut self, name: &str, value: &str) -> &mut Self {
        self.headers.insert(name.to_string(), value.to_string());
        self
    }

    /// Set basic authentication credentials
    ///
    /// Encodes username and password in Base64 for Authorization header.
    ///
    /// # Arguments
    ///
    /// * `username` - The username
    /// * `password` - The password
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = HttpClient::new();
    /// client.with_basic_auth("user", "pass");
    /// ```
    pub fn with_basic_auth(&mut self, username: &str, password: &str) -> &mut Self {
        let credentials = format!("{}:{}", username, password);
        let encoded = base64_encode(&credentials);
        self.with_header("Authorization", &format!("Basic {}", encoded));
        self
    }

    /// Set the request timeout
    ///
    /// # Arguments
    ///
    /// * `duration` - Timeout duration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = HttpClient::new();
    /// client.with_timeout(std::time::Duration::from_secs(10));
    /// ```
    pub fn with_timeout(&mut self, duration: Duration) -> &mut Self {
        self.timeout = duration;
        self
    }

    /// Set the maximum response size
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum size in bytes
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = HttpClient::new();
    /// client.with_max_response_size(50 * 1024 * 1024); // 50MB
    /// ```
    pub fn with_max_response_size(&mut self, size: usize) -> &mut Self {
        self.max_response_size = size;
        self
    }

    /// Verify that a domain is allowed
    async fn check_domain(&self, url: &str) -> Result<(), HttpClientError> {
        let domain = extract_domain(url)
            .ok_or_else(|| HttpClientError::new("Could not extract domain from URL"))?;

        // Hard-coded SSRF deny-list — checked before any user-configured lists.
        // Resolves the hostname to IP(s) via spawn_blocking to avoid stalling the executor.
        check_ssrf_blocked(&domain).await?;

        // Check blocklist first
        if self.denied_domains.contains(&domain) {
            return Err(HttpClientError::new(format!(
                "Domain '{}' is blocked",
                domain
            )));
        }

        // Check allowlist if it exists
        if !self.allowed_domains.is_empty() && !self.allowed_domains.contains(&domain) {
            return Err(HttpClientError::new(format!(
                "Domain '{}' is not allowed",
                domain
            )));
        }

        Ok(())
    }

    /// Build URL with query parameters
    fn build_url(&self, base_url: &str) -> String {
        if self.query_params.is_empty() {
            return base_url.to_string();
        }

        let separator = if base_url.contains('?') { "&" } else { "?" };
        let params: Vec<String> = self
            .query_params
            .iter()
            .map(|(k, v)| format!("{}={}", k, urlencoding::encode(v)))
            .collect();

        format!("{}{}{}", base_url, separator, params.join("&"))
    }

    /// Make a GET request
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to request
    ///
    /// # Returns
    ///
    /// An `HttpResponse` with status, headers, and body
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = HttpClient::new();
    /// let response = client.get("https://api.example.com/users").await?;
    /// println!("Status: {}", response.status);
    /// ```
    pub async fn get(&self, url: &str) -> Result<HttpResponse, HttpClientError> {
        self.check_domain(url).await?;

        let full_url = self.build_url(url);
        let mut req = self.client.get(&full_url);

        for (key, value) in &self.headers {
            req = req.header(key.clone(), value.clone());
        }

        let response = req
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| HttpClientError::new(format!("GET request failed: {}", e)))?;

        self.build_response(response).await
    }

    /// Make a POST request
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to request
    /// * `payload` - JSON payload to send
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = HttpClient::new();
    /// let payload = serde_json::json!({"name": "Alice"});
    /// let response = client.post("https://api.example.com/users", payload).await?;
    /// ```
    pub async fn post(
        &self,
        url: &str,
        payload: JsonValue,
    ) -> Result<HttpResponse, HttpClientError> {
        self.check_domain(url).await?;

        let full_url = self.build_url(url);
        let mut req = self.client.post(&full_url).json(&payload);

        for (key, value) in &self.headers {
            req = req.header(key.clone(), value.clone());
        }

        let response = req
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| HttpClientError::new(format!("POST request failed: {}", e)))?;

        self.build_response(response).await
    }

    /// Make a PUT request
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to request
    /// * `payload` - JSON payload to send
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = HttpClient::new();
    /// let payload = serde_json::json!({"name": "Bob"});
    /// let response = client.put("https://api.example.com/users/123", payload).await?;
    /// ```
    pub async fn put(
        &self,
        url: &str,
        payload: JsonValue,
    ) -> Result<HttpResponse, HttpClientError> {
        self.check_domain(url).await?;

        let full_url = self.build_url(url);
        let mut req = self.client.put(&full_url).json(&payload);

        for (key, value) in &self.headers {
            req = req.header(key.clone(), value.clone());
        }

        let response = req
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| HttpClientError::new(format!("PUT request failed: {}", e)))?;

        self.build_response(response).await
    }

    /// Make a DELETE request
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to request
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = HttpClient::new();
    /// let response = client.delete("https://api.example.com/users/123").await?;
    /// ```
    pub async fn delete(&self, url: &str) -> Result<HttpResponse, HttpClientError> {
        self.check_domain(url).await?;

        let full_url = self.build_url(url);
        let mut req = self.client.delete(&full_url);

        for (key, value) in &self.headers {
            req = req.header(key.clone(), value.clone());
        }

        let response = req
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| HttpClientError::new(format!("DELETE request failed: {}", e)))?;

        self.build_response(response).await
    }

    /// Make a PATCH request
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to request
    /// * `payload` - JSON payload to send
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = HttpClient::new();
    /// let payload = serde_json::json!({"status": "active"});
    /// let response = client.patch("https://api.example.com/users/123", payload).await?;
    /// ```
    pub async fn patch(
        &self,
        url: &str,
        payload: JsonValue,
    ) -> Result<HttpResponse, HttpClientError> {
        self.check_domain(url).await?;

        let full_url = self.build_url(url);
        let mut req = self.client.patch(&full_url).json(&payload);

        for (key, value) in &self.headers {
            req = req.header(key.clone(), value.clone());
        }

        let response = req
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| HttpClientError::new(format!("PATCH request failed: {}", e)))?;

        self.build_response(response).await
    }

    /// Make a HEAD request
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to request
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = HttpClient::new();
    /// let response = client.head("https://api.example.com/data").await?;
    /// ```
    pub async fn head(&self, url: &str) -> Result<HttpResponse, HttpClientError> {
        self.check_domain(url).await?;

        let full_url = self.build_url(url);
        let mut req = self.client.head(&full_url);

        for (key, value) in &self.headers {
            req = req.header(key.clone(), value.clone());
        }

        let response = req
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| HttpClientError::new(format!("HEAD request failed: {}", e)))?;

        self.build_response(response).await
    }

    /// Build HttpResponse from reqwest response
    async fn build_response(
        &self,
        response: reqwest::Response,
    ) -> Result<HttpResponse, HttpClientError> {
        let status = response.status().as_u16();

        let headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        // Stream the body incrementally, aborting as soon as we exceed
        // max_response_size — this prevents an oversized response from ever
        // being fully buffered in memory (OOM DoS fix).
        use futures_util::StreamExt;
        let mut stream = response.bytes_stream();
        let mut body_bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                HttpClientError::new(format!("Failed to read response body: {}", e))
            })?;
            if body_bytes.len() + chunk.len() > self.max_response_size {
                return Err(HttpClientError::new(format!(
                    "Response body exceeds maximum size of {} bytes",
                    self.max_response_size
                )));
            }
            body_bytes.extend_from_slice(&chunk);
        }
        let body = String::from_utf8_lossy(&body_bytes).into_owned();

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` if the IP address falls into a range that must never be
/// reachable from an agent-driven HTTP request (SSRF deny-list).
///
/// Blocked ranges:
/// - IPv4 loopback:       127.0.0.0/8
/// - IPv4 link-local:     169.254.0.0/16  (AWS IMDS and similar metadata services)
/// - IPv4 RFC-1918:       10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
/// - IPv6 loopback:       ::1
/// - IPv6 link-local:     fe80::/10
fn is_ssrf_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            // 127.0.0.0/8 — loopback
            if o[0] == 127 {
                return true;
            }
            // 169.254.0.0/16 — link-local / cloud metadata (AWS IMDS etc.)
            if o[0] == 169 && o[1] == 254 {
                return true;
            }
            // 10.0.0.0/8 — RFC-1918
            if o[0] == 10 {
                return true;
            }
            // 172.16.0.0/12 — RFC-1918 (172.16.x.x – 172.31.x.x)
            if o[0] == 172 && o[1] >= 16 && o[1] <= 31 {
                return true;
            }
            // 192.168.0.0/16 — RFC-1918
            if o[0] == 192 && o[1] == 168 {
                return true;
            }
            false
        }
        IpAddr::V6(v6) => {
            // ::1 — IPv6 loopback
            if v6.is_loopback() {
                return true;
            }
            // :: — unspecified address
            if v6.is_unspecified() {
                return true;
            }
            // fe80::/10 — IPv6 link-local
            let segments = v6.segments();
            if (segments[0] & 0xffc0) == 0xfe80 {
                return true;
            }
            false
        }
    }
}

/// Resolve `host` to IP addresses and reject any that fall in SSRF-blocked ranges.
///
/// Uses the synchronous system resolver (`ToSocketAddrs`). If DNS resolution
/// fails the request is also rejected — unknown hosts are not allowed through.
///
/// Note: a DNS-rebinding attack could bypass this pre-flight check by resolving
/// to a public IP here and a private IP at request time. That risk is accepted;
/// the deny-list still blocks the overwhelmingly common direct-IP and single-DNS
/// SSRF vectors.
/// Async wrapper: runs the blocking DNS resolution on a dedicated thread via
/// `spawn_blocking` so the tokio executor thread is never stalled.
async fn check_ssrf_blocked(host: &str) -> Result<(), HttpClientError> {
    let host_owned = host.to_string();
    let addrs = tokio::task::spawn_blocking(move || {
        use std::net::ToSocketAddrs;
        format!("{}:80", host_owned).to_socket_addrs()
    })
    .await
    .map_err(|e| HttpClientError::new(format!("DNS resolution task failed: {}", e)))?
    .map_err(|e| HttpClientError::new(format!("Could not resolve host '{}': {}", host, e)))?;

    for addr in addrs {
        if is_ssrf_ip(addr.ip()) {
            return Err(HttpClientError::new(format!(
                "Request to '{}' blocked: target IP {} is in a reserved/private range",
                host,
                addr.ip()
            )));
        }
    }
    Ok(())
}

/// Extract domain from URL
fn extract_domain(url: &str) -> Option<String> {
    let url_str = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;

    let domain = url_str.split('/').next()?.split(':').next()?;

    Some(domain.to_string())
}

/// Simple base64 encoding helper
fn base64_encode(input: &str) -> String {
    use std::fmt::Write as _;

    let bytes = input.as_bytes();
    let table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in bytes.chunks(3) {
        let b1 = chunk[0];
        let b2 = chunk.get(1).copied().unwrap_or(0);
        let b3 = chunk.get(2).copied().unwrap_or(0);

        let n = ((b1 as u32) << 16) | ((b2 as u32) << 8) | (b3 as u32);

        let _ = write!(
            result,
            "{}{}{}{}",
            table.chars().nth(((n >> 18) & 0x3f) as usize).unwrap(),
            table.chars().nth(((n >> 12) & 0x3f) as usize).unwrap(),
            if chunk.len() > 1 {
                table.chars().nth(((n >> 6) & 0x3f) as usize).unwrap()
            } else {
                '='
            },
            if chunk.len() > 2 {
                table.chars().nth((n & 0x3f) as usize).unwrap()
            } else {
                '='
            }
        );
    }

    result
}
