//! HTTP Client Pool for maintaining persistent connections per base URL.
//!
//! This module provides a singleton pool of reqwest::Client instances, one per base URL.
//! This ensures that:
//! - HTTP connections are reused across multiple requests (connection pooling)
//! - DNS lookups are minimized
//! - TLS handshakes are reused where possible
//! - TCP connections are kept alive to avoid reconnection overhead
//!
//! The reqwest::Client is configured with optimal settings for persistent connections:
//! - `pool_idle_timeout`: Keeps idle connections alive for 90 seconds
//! - `pool_max_idle_per_host`: Allows up to 10 idle connections per host
//! - `tcp_keepalive`: Sends keepalive packets every 60 seconds to prevent connection closure

use once_cell::sync::Lazy;
use reqwest;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

/// Global HTTP client pool, lazily initialized on first access.
static HTTP_CLIENT_POOL: Lazy<Mutex<HashMap<String, reqwest::Client>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Get or create a shared HTTP client for the given base URL.
///
/// This function maintains a singleton pool of reqwest::Client instances.
/// Each base URL gets its own client to ensure proper connection pooling.
///
/// # Arguments
/// * `base_url` - The base URL for which to get/create an HTTP client
///
/// # Returns
/// A cloned reqwest::Client configured for persistent connections
pub fn get_http_client(base_url: &str) -> reqwest::Client {
    let mut pool = HTTP_CLIENT_POOL.lock().unwrap();
    
    if let Some(client) = pool.get(base_url) {
        return client.clone();
    }
    
    // Create a new client with optimal settings for persistent connections
    let client = reqwest::ClientBuilder::new()
        // Keep idle connections alive for 90 seconds
        .pool_idle_timeout(Some(Duration::from_secs(90)))
        // Allow up to 10 idle connections per host for better throughput
        .pool_max_idle_per_host(10)
        // Enable TCP keepalive to prevent connection drops
        .tcp_keepalive(Some(Duration::from_secs(60)))
        // Set a reasonable timeout for the entire request
        .timeout(Duration::from_secs(300))
        .build()
        .expect("Failed to build HTTP client");
    
    pool.insert(base_url.to_string(), client.clone());
    client
}
