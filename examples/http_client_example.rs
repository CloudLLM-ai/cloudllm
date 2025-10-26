//! Example demonstrating the HTTP Client tool with comprehensive usage patterns
//!
//! This example shows how to use the HTTP Client for various operations:
//! - Making HTTP requests (GET, POST, PUT, DELETE, PATCH)
//! - Setting headers and query parameters
//! - Authentication (basic auth, bearer tokens)
//! - Domain allowlist/blocklist for security
//! - Timeout and response size configuration
//! - JSON response parsing
//! - Error handling

use cloudllm::tools::HttpClient;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CloudLLM HTTP Client Tool Example ===\n");

    // ===== CLIENT CREATION & CONFIGURATION =====
    println!("--- Client Creation & Configuration ---");
    demo_client_setup().await?;

    // ===== DOMAIN SECURITY =====
    println!("\n--- Domain Security (Allowlist/Blocklist) ---");
    demo_domain_security().await?;

    // ===== HEADERS & AUTHENTICATION =====
    println!("\n--- Headers & Authentication ---");
    demo_headers_auth().await?;

    // ===== QUERY PARAMETERS =====
    println!("\n--- Query Parameters ---");
    demo_query_parameters().await?;

    // ===== TIMEOUT & SIZE LIMITS =====
    println!("\n--- Timeout & Size Limit Configuration ---");
    demo_limits().await?;

    // ===== BUILDER PATTERN =====
    println!("\n--- Builder Pattern (Chainable Configuration) ---");
    demo_builder_pattern().await?;

    // ===== ERROR HANDLING =====
    println!("\n--- Error Handling ---");
    demo_error_handling().await?;

    println!("\n✓ All examples completed successfully!");
    Ok(())
}

async fn demo_client_setup() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating HTTP client with default settings...");
    let _client1 = HttpClient::new();
    println!("  ✓ Client created successfully");

    println!("  Creating client with default constructor...");
    let _client2 = HttpClient::default();
    println!("  ✓ Default client created successfully");

    println!("  Both constructors work identically for basic usage");
    Ok(())
}

async fn demo_domain_security() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Example 1: Allowlist - restrict to trusted domains");
    let mut client = HttpClient::new();
    client.allow_domain("api.example.com");
    client.allow_domain("api.partner.com");
    println!("  ✓ Allowed domains: api.example.com, api.partner.com");

    println!("\n  Example 2: Blocklist - prevent access to malicious domains");
    let mut client = HttpClient::new();
    client.deny_domain("malicious.com");
    client.deny_domain("phishing.net");
    println!("  ✓ Blocked domains: malicious.com, phishing.net");

    println!("\n  Example 3: Mixed - allowlist with additional blocklist");
    let mut client = HttpClient::new();
    client.allow_domain("trusted-api.com");
    client.deny_domain("evil.trusted-api.com"); // Block specific subdomain
    println!("  ✓ Allowed: trusted-api.com (except evil.trusted-api.com)");

    println!("\n  Example 4: Blocklist takes precedence over allowlist");
    let mut client = HttpClient::new();
    client.allow_domain("api.example.com");
    client.deny_domain("api.example.com"); // Block it anyway
    println!("  ✓ api.example.com is blocked (blocklist takes precedence)");

    println!("\n  Example 5: Empty allowlist allows all (only blocklist checked)");
    let _client = HttpClient::new();
    println!("  ✓ No allowlist set, all domains allowed (except blocklist)");

    Ok(())
}

async fn demo_headers_auth() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Example 1: Custom headers");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_header("X-API-Version", "v2");
    client.with_header("X-Request-ID", "12345-67890");
    client.with_header("Accept", "application/json");
    println!("  ✓ Added custom headers:");
    println!("    - X-API-Version: v2");
    println!("    - X-Request-ID: 12345-67890");
    println!("    - Accept: application/json");

    println!("\n  Example 2: Bearer token authentication");
    let mut client = HttpClient::new();
    client.allow_domain("api.example.com");
    client.with_header("Authorization", "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...");
    println!("  ✓ Added Bearer token authentication header");

    println!("\n  Example 3: Basic authentication");
    let mut client = HttpClient::new();
    client.allow_domain("api.example.com");
    client.with_basic_auth("username", "password");
    println!("  ✓ Configured basic auth (username: username)");
    println!("    Base64 encoded internally for transmission");

    println!("\n  Example 4: Multiple headers chained");
    let mut client = HttpClient::new();
    client
        .allow_domain("api.example.com")
        .with_header("Content-Type", "application/json")
        .with_header("User-Agent", "CloudLLM/1.0")
        .with_header("Accept-Language", "en-US");
    println!("  ✓ Multiple headers configured via builder pattern");

    Ok(())
}

async fn demo_query_parameters() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Example 1: Single query parameter");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_query_param("key", "value");
    println!("  ✓ Query param: key=value");

    println!("\n  Example 2: Multiple query parameters");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_query_param("page", "1");
    client.with_query_param("limit", "50");
    client.with_query_param("sort", "created_at");
    println!("  ✓ Query params: page=1&limit=50&sort=created_at");

    println!("\n  Example 3: Special characters (automatic URL encoding)");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_query_param("search", "hello world");
    client.with_query_param("filter", "status=active&priority=high");
    println!("  ✓ Special characters automatically URL-encoded");
    println!("    - 'hello world' → 'hello%20world'");
    println!("    - '&' → '%26' (within param values)");

    Ok(())
}

async fn demo_limits() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Example 1: Timeout configuration");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_timeout(Duration::from_secs(30));
    println!("  ✓ Request timeout set to 30 seconds");

    println!("\n  Example 2: Short timeout for quick operations");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_timeout(Duration::from_secs(5));
    println!("  ✓ Quick timeout set to 5 seconds");

    println!("\n  Example 3: Maximum response size");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_max_response_size(50 * 1024 * 1024); // 50MB
    println!("  ✓ Max response size: 50 MB");

    println!("\n  Example 4: Small response limit (for safety)");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    client.with_max_response_size(1024 * 1024); // 1MB
    println!("  ✓ Max response size: 1 MB (conservative limit)");

    println!("\n  Example 5: Practical configuration");
    let mut client = HttpClient::new();
    client.allow_domain("api.example.com");
    client.with_timeout(Duration::from_secs(10));
    client.with_max_response_size(10 * 1024 * 1024); // 10MB
    println!("  ✓ Practical config: 10s timeout, 10MB limit");

    Ok(())
}

async fn demo_builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Example 1: Chain multiple configurations");
    let mut client = HttpClient::new();
    let _result = client
        .allow_domain("api.example.com")
        .with_header("Authorization", "Bearer token123")
        .with_query_param("format", "json")
        .with_timeout(Duration::from_secs(15));
    println!("  ✓ Chained configuration:");
    println!("    - Domain: api.example.com");
    println!("    - Auth: Bearer token");
    println!("    - Query: format=json");
    println!("    - Timeout: 15 seconds");

    println!("\n  Example 2: Complex chain for REST API");
    let mut client = HttpClient::new();
    let _result = client
        .allow_domain("api.service.com")
        .with_basic_auth("api_user", "api_key")
        .with_header("Accept", "application/json")
        .with_header("User-Agent", "CloudLLM-Client/1.0")
        .with_query_param("v", "2")
        .with_max_response_size(5 * 1024 * 1024);
    println!("  ✓ Complex REST API configuration:");
    println!("    - Domain: api.service.com");
    println!("    - Auth: Basic auth");
    println!("    - Headers: Accept, User-Agent");
    println!("    - Query: v=2");
    println!("    - Size limit: 5 MB");

    println!("\n  Example 3: Reconfigure by setting new values");
    let mut client = HttpClient::new();
    client.with_timeout(Duration::from_secs(30));
    println!("  ✓ Initial timeout: 30 seconds");
    client.with_timeout(Duration::from_secs(10));
    println!("  ✓ Updated timeout: 10 seconds (new value replaces old)");

    Ok(())
}

async fn demo_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Example 1: Blocked domain returns error");
    let mut client = HttpClient::new();
    client.deny_domain("blocked.example.com");
    let result = client.get("https://blocked.example.com/api/data").await;
    match result {
        Err(e) => println!("  ✓ Expected error: {}", e),
        Ok(_) => println!("  ✗ Unexpected success"),
    }

    println!("\n  Example 2: Invalid URL handling");
    let mut client = HttpClient::new();
    client.allow_domain("example.com");
    let result = client.get("not-a-valid-url").await;
    match result {
        Err(e) => println!("  ✓ Expected error for malformed URL: {}", e),
        Ok(_) => println!("  ✗ Unexpected success"),
    }

    println!("\n  Example 3: Graceful handling in application");
    let mut client = HttpClient::new();
    client.allow_domain("api.example.com");
    match client.get("https://api.example.com/endpoint").await {
        Ok(response) => {
            println!("  ✓ Request succeeded");
            println!("    Status: {} ({})",
                response.status,
                if response.is_success() { "Success" }
                else if response.is_client_error() { "Client Error" }
                else { "Server Error" }
            );
        }
        Err(e) => println!("  ℹ Request failed: {} (may be network error)", e),
    }

    println!("\n  Example 4: JSON response error handling");
    let mut client = HttpClient::new();
    client.allow_domain("api.example.com");
    match client.get("https://api.example.com/data").await {
        Ok(response) => match response.json() {
            Ok(json_data) => println!("  ✓ Parsed JSON: {}", json_data),
            Err(_) => println!("  ✓ Response is not valid JSON"),
        },
        Err(e) => println!("  ℹ Request error: {}", e),
    }

    Ok(())
}
