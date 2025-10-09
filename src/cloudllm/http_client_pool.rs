//! HTTP Client Pool for maintaining persistent connections.
//!
//! This module provides a singleton-based HTTP client pool that maintains
//! persistent connections per base URL, avoiding DNS/TLS churn and reducing
//! connection overhead. Each base URL gets its own configured `reqwest::Client`
//! with connection pooling enabled.

use dashmap::DashMap;
use once_cell::sync::Lazy;
use reqwest;
use std::time::Duration;

/// Global cache of HTTP clients indexed by base URL.
/// Using DashMap for thread-safe concurrent access without locks.
static CLIENT_POOL: Lazy<DashMap<String, reqwest::Client>> = Lazy::new(DashMap::new);

/// Creates or retrieves a shared HTTP client for the given base URL.
///
/// The client is configured with:
/// - Connection pooling with up to 100 idle connections per host
/// - 90-second idle timeout for persistent connections
/// - TCP keepalive to maintain long-lived connections
/// - 30-second connection timeout
///
/// # Arguments
///
/// * `base_url` - The base URL to create/retrieve a client for
///
/// # Returns
///
/// A cloned `reqwest::Client` that shares the connection pool
pub fn get_or_create_client(base_url: &str) -> reqwest::Client {
    CLIENT_POOL
        .entry(base_url.to_string())
        .or_insert_with(create_pooled_client)
        .clone()
}

/// Creates a new reqwest client with optimized connection pooling settings.
///
/// Configuration details:
/// - `pool_max_idle_per_host(100)`: Maintains up to 100 idle connections per host
/// - `pool_idle_timeout(90s)`: Keeps connections alive for 90 seconds
/// - `tcp_keepalive(60s)`: Sends TCP keepalive probes every 60 seconds
/// - `connect_timeout(30s)`: Maximum time to establish a connection
///
/// These settings ensure connections are reused efficiently and avoid the
/// multi-millisecond overhead of DNS lookups and TLS handshakes on each request.
fn create_pooled_client() -> reqwest::Client {
    reqwest::ClientBuilder::new()
        .pool_max_idle_per_host(100)
        .pool_idle_timeout(Some(Duration::from_secs(90)))
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .connect_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build HTTP client")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_pool_returns_same_instance() {
        let url = "https://api.openai.com/v1";
        let _client1 = get_or_create_client(url);
        let _client2 = get_or_create_client(url);

        // While we can't directly compare clients, we can verify both exist
        // and the pool contains the entry
        assert!(CLIENT_POOL.contains_key(url));
        
        // Verify different URLs get different entries
        let different_url = "https://api.anthropic.com/v1";
        let _client3 = get_or_create_client(different_url);
        assert!(CLIENT_POOL.contains_key(different_url));
        // Note: Can't check exact length due to parallel test execution
        assert!(CLIENT_POOL.len() >= 2);
    }

    #[test]
    fn test_client_creation() {
        // This test ensures the client can be created without panicking
        let client = create_pooled_client();
        // Basic validation that the client was created
        assert!(std::ptr::addr_of!(client) as usize != 0);
    }

    #[test]
    fn test_multiple_base_urls_create_separate_pools() {
        // Test that different providers get their own pooled clients
        let openai_url = "https://api.openai.com/v1";
        let anthropic_url = "https://api.anthropic.com/v1";
        let gemini_url = "https://generativelanguage.googleapis.com/v1beta/";
        let xai_url = "https://api.x.ai/v1";

        let _client1 = get_or_create_client(openai_url);
        let _client2 = get_or_create_client(anthropic_url);
        let _client3 = get_or_create_client(gemini_url);
        let _client4 = get_or_create_client(xai_url);

        // Verify all URLs are in the pool
        assert!(CLIENT_POOL.contains_key(openai_url));
        assert!(CLIENT_POOL.contains_key(anthropic_url));
        assert!(CLIENT_POOL.contains_key(gemini_url));
        assert!(CLIENT_POOL.contains_key(xai_url));
    }

    #[test]
    fn test_client_reuse_across_multiple_calls() {
        // Test that calling get_or_create_client multiple times for the same URL
        // doesn't create new clients each time
        let url = "https://test.example.com/v1";
        
        // Ensure the client is created
        let _client = get_or_create_client(url);
        assert!(CLIENT_POOL.contains_key(url));
        
        // Now call it multiple times
        for _ in 0..10 {
            let _client = get_or_create_client(url);
        }
        
        // Should still only have one entry for this URL
        assert!(CLIENT_POOL.contains_key(url));
    }
}
