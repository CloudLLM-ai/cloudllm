//! Utilities for MCPServerBuilder
//!
//! This module provides helper functionality for the MCP server builder,
//! including IP filtering, authentication, and CIDR matching.

use sha2::{Digest, Sha256};
use std::net::IpAddr;
use std::str::FromStr;
use subtle::ConstantTimeEq;

/// IP filter for restricting server access
#[derive(Debug, Clone)]
pub struct IpFilter {
    /// List of allowed IP addresses and CIDR blocks
    allowed: Vec<IpFilterEntry>,
}

#[derive(Debug, Clone)]
enum IpFilterEntry {
    /// Single IP address
    Single(IpAddr),
    /// CIDR block (network/prefix_length)
    Cidr { network: IpAddr, prefix_len: u8 },
}

impl IpFilter {
    /// Create a new empty IP filter (allows all)
    pub fn new() -> Self {
        Self {
            allowed: Vec::new(),
        }
    }

    /// Add an allowed IP address or CIDR block
    ///
    /// # Arguments
    ///
    /// * `ip_or_cidr` - Either an IP address (e.g., "127.0.0.1", "::1") or a CIDR block
    ///   (e.g., "192.168.1.0/24", "2001:db8::/32")
    ///
    /// # Returns
    ///
    /// `Err` if the input is invalid
    pub fn allow(&mut self, ip_or_cidr: &str) -> Result<(), String> {
        // Try parsing as CIDR first
        if let Some(slash_pos) = ip_or_cidr.find('/') {
            let (network_part, prefix_part) = ip_or_cidr.split_at(slash_pos);
            let prefix_str = &prefix_part[1..]; // Skip the '/'

            let network = IpAddr::from_str(network_part)
                .map_err(|e| format!("Invalid network address: {}", e))?;

            let prefix_len: u8 = prefix_str
                .parse()
                .map_err(|_| format!("Invalid CIDR prefix length: {}", prefix_str))?;

            // Validate prefix length based on IP type
            let max_prefix = match network {
                IpAddr::V4(_) => 32,
                IpAddr::V6(_) => 128,
            };

            if prefix_len > max_prefix {
                return Err(format!(
                    "CIDR prefix length {} exceeds maximum {} for {:?}",
                    prefix_len, max_prefix, network
                ));
            }

            self.allowed.push(IpFilterEntry::Cidr {
                network,
                prefix_len,
            });
            Ok(())
        } else {
            // Try parsing as single IP
            let ip =
                IpAddr::from_str(ip_or_cidr).map_err(|e| format!("Invalid IP address: {}", e))?;
            self.allowed.push(IpFilterEntry::Single(ip));
            Ok(())
        }
    }

    /// Check if an IP address is allowed
    pub fn is_allowed(&self, ip: IpAddr) -> bool {
        // If no restrictions, allow all
        if self.allowed.is_empty() {
            return true;
        }

        // Check each allowed entry
        self.allowed.iter().any(|entry| self.matches(ip, entry))
    }

    /// Check if an IP matches a filter entry
    fn matches(&self, ip: IpAddr, entry: &IpFilterEntry) -> bool {
        match entry {
            IpFilterEntry::Single(allowed_ip) => ip == *allowed_ip,
            IpFilterEntry::Cidr {
                network,
                prefix_len,
            } => self.ip_in_cidr(ip, *network, *prefix_len),
        }
    }

    /// Check if an IP is in a CIDR block
    fn ip_in_cidr(&self, ip: IpAddr, network: IpAddr, prefix_len: u8) -> bool {
        match (ip, network) {
            (IpAddr::V4(ip), IpAddr::V4(net)) => {
                let ip_bits = u32::from(ip);
                let net_bits = u32::from(net);
                let mask = if prefix_len == 0 {
                    0
                } else {
                    0xFFFFFFFFu32 << (32 - prefix_len)
                };
                (ip_bits & mask) == (net_bits & mask)
            }
            (IpAddr::V6(ip), IpAddr::V6(net)) => {
                let ip_bits = u128::from(ip);
                let net_bits = u128::from(net);
                let mask = if prefix_len == 0 {
                    0
                } else {
                    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFu128 << (128 - prefix_len)
                };
                (ip_bits & mask) == (net_bits & mask)
            }
            _ => false, // IPv4 vs IPv6 mismatch
        }
    }
}

impl Default for IpFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Authentication configuration for MCP server
#[derive(Debug, Clone)]
pub enum AuthConfig {
    /// No authentication required
    None,
    /// Bearer token authentication
    Bearer(String),
    /// Basic authentication (username:password base64 encoded)
    Basic { username: String, password: String },
}

impl AuthConfig {
    /// Create bearer token authentication
    pub fn bearer(token: impl Into<String>) -> Self {
        Self::Bearer(token.into())
    }

    /// Create basic authentication
    pub fn basic(username: impl Into<String>, password: impl Into<String>) -> Self {
        Self::Basic {
            username: username.into(),
            password: password.into(),
        }
    }

    /// Validate an Authorization header
    ///
    /// # Arguments
    ///
    /// * `header` - The Authorization header value (e.g., "Bearer token123")
    ///
    /// # Returns
    ///
    /// `true` if the header matches the configured authentication
    pub fn validate(&self, header: &str) -> bool {
        match self {
            AuthConfig::None => true,
            AuthConfig::Bearer(token) => {
                if let Some(token_part) = header.strip_prefix("Bearer ") {
                    // subtle::ConstantTimeEq on SHA-256 digests prevents timing oracle attacks.
                    // The optimizer cannot short-circuit ct_eq() the way it can with `==`.
                    let expected_hash = Sha256::digest(token.as_bytes());
                    let provided_hash = Sha256::digest(token_part.as_bytes());
                    expected_hash.ct_eq(&provided_hash).into()
                } else {
                    false
                }
            }
            AuthConfig::Basic { username, password } => {
                if let Some(creds_part) = header.strip_prefix("Basic ") {
                    // Decode base64 and check against username:password
                    if let Ok(decoded) = base64_decode(creds_part) {
                        let expected = format!("{}:{}", username, password);
                        // subtle::ConstantTimeEq prevents timing oracle on credentials.
                        let expected_hash = Sha256::digest(expected.as_bytes());
                        let decoded_hash = Sha256::digest(decoded.as_bytes());
                        expected_hash.ct_eq(&decoded_hash).into()
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }
}

/// Decode base64 string
fn base64_decode(s: &str) -> Result<String, String> {
    // Simple base64 decoding without external dependencies
    const BASE64_TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [255u8; 256];
    for (i, &c) in BASE64_TABLE.iter().enumerate() {
        table[c as usize] = i as u8;
    }

    let input = s.trim_end_matches('=');
    let mut output = Vec::new();
    let bytes = input.as_bytes();

    for chunk in bytes.chunks(4) {
        if chunk.len() < 2 {
            break;
        }

        let mut buf = [0u8; 4];
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' {
                break;
            }
            buf[i] = table[c as usize];
            if buf[i] == 255 {
                return Err("Invalid base64 character".to_string());
            }
        }

        let b1 = (buf[0] << 2) | (buf[1] >> 4);
        output.push(b1);

        if chunk.len() > 2 && chunk[2] != b'=' {
            let b2 = ((buf[1] & 0x0F) << 4) | (buf[2] >> 2);
            output.push(b2);
        }

        if chunk.len() > 3 && chunk[3] != b'=' {
            let b3 = ((buf[2] & 0x03) << 6) | buf[3];
            output.push(b3);
        }
    }

    String::from_utf8(output).map_err(|e| e.to_string())
}
