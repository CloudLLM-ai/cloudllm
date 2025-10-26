//! Tests for MCP Server Builder Utilities
//!
//! Tests IP filtering, CIDR matching, and authentication validation

use cloudllm::cloudllm::mcp_server_builder_utils::{AuthConfig, IpFilter};

// ============= IP Filter Tests =============

#[test]
fn test_single_ip_filter() {
    let mut filter = IpFilter::new();
    filter.allow("127.0.0.1").unwrap();

    assert!(filter.is_allowed("127.0.0.1".parse().unwrap()));
    assert!(!filter.is_allowed("192.168.1.1".parse().unwrap()));
}

#[test]
fn test_ipv6_single() {
    let mut filter = IpFilter::new();
    filter.allow("::1").unwrap();

    assert!(filter.is_allowed("::1".parse().unwrap()));
    assert!(!filter.is_allowed("::2".parse().unwrap()));
}

#[test]
fn test_ipv4_cidr() {
    let mut filter = IpFilter::new();
    filter.allow("192.168.1.0/24").unwrap();

    assert!(filter.is_allowed("192.168.1.0".parse().unwrap()));
    assert!(filter.is_allowed("192.168.1.1".parse().unwrap()));
    assert!(filter.is_allowed("192.168.1.255".parse().unwrap()));
    assert!(!filter.is_allowed("192.168.2.1".parse().unwrap()));
}

#[test]
fn test_ipv6_cidr() {
    let mut filter = IpFilter::new();
    filter.allow("2001:db8::/32").unwrap();

    assert!(filter.is_allowed("2001:db8::1".parse().unwrap()));
    assert!(filter.is_allowed("2001:db8:1234:5678::1".parse().unwrap()));
    assert!(!filter.is_allowed("2001:db9::1".parse().unwrap()));
}

#[test]
fn test_multiple_entries() {
    let mut filter = IpFilter::new();
    filter.allow("127.0.0.1").unwrap();
    filter.allow("::1").unwrap();
    filter.allow("192.168.0.0/16").unwrap();

    assert!(filter.is_allowed("127.0.0.1".parse().unwrap()));
    assert!(filter.is_allowed("::1".parse().unwrap()));
    assert!(filter.is_allowed("192.168.1.1".parse().unwrap()));
    assert!(!filter.is_allowed("10.0.0.1".parse().unwrap()));
}

#[test]
fn test_empty_filter_allows_all() {
    let filter = IpFilter::new();
    assert!(filter.is_allowed("127.0.0.1".parse().unwrap()));
    assert!(filter.is_allowed("8.8.8.8".parse().unwrap()));
}

#[test]
fn test_invalid_cidr_prefix_too_large() {
    let mut filter = IpFilter::new();
    assert!(filter.allow("192.168.1.0/33").is_err());
}

#[test]
fn test_invalid_cidr_ipv6_prefix_too_large() {
    let mut filter = IpFilter::new();
    assert!(filter.allow("2001:db8::/129").is_err());
}

#[test]
fn test_invalid_ip_address() {
    let mut filter = IpFilter::new();
    assert!(filter.allow("invalid/24").is_err());
}

#[test]
fn test_ipv4_cidr_boundary_24bit() {
    let mut filter = IpFilter::new();
    filter.allow("10.0.0.0/24").unwrap();

    // Should allow 10.0.0.0 - 10.0.0.255
    assert!(filter.is_allowed("10.0.0.0".parse().unwrap()));
    assert!(filter.is_allowed("10.0.0.127".parse().unwrap()));
    assert!(filter.is_allowed("10.0.0.255".parse().unwrap()));

    // Should not allow 10.0.1.0
    assert!(!filter.is_allowed("10.0.1.0".parse().unwrap()));
}

#[test]
fn test_ipv4_cidr_boundary_16bit() {
    let mut filter = IpFilter::new();
    filter.allow("172.16.0.0/16").unwrap();

    // Should allow 172.16.x.x
    assert!(filter.is_allowed("172.16.0.0".parse().unwrap()));
    assert!(filter.is_allowed("172.16.255.255".parse().unwrap()));

    // Should not allow 172.17.0.0
    assert!(!filter.is_allowed("172.17.0.0".parse().unwrap()));
}

#[test]
fn test_ipv4_cidr_32bit_single() {
    let mut filter = IpFilter::new();
    filter.allow("8.8.8.8/32").unwrap();

    assert!(filter.is_allowed("8.8.8.8".parse().unwrap()));
    assert!(!filter.is_allowed("8.8.8.9".parse().unwrap()));
}

#[test]
fn test_ipv4_cidr_0bit_all() {
    let mut filter = IpFilter::new();
    filter.allow("0.0.0.0/0").unwrap();

    // Should allow any IPv4
    assert!(filter.is_allowed("0.0.0.0".parse().unwrap()));
    assert!(filter.is_allowed("127.0.0.1".parse().unwrap()));
    assert!(filter.is_allowed("192.168.1.1".parse().unwrap()));
    assert!(filter.is_allowed("255.255.255.255".parse().unwrap()));
}

// ============= Authentication Tests =============

#[test]
fn test_bearer_auth_valid() {
    let auth = AuthConfig::bearer("secret123");
    assert!(auth.validate("Bearer secret123"));
}

#[test]
fn test_bearer_auth_invalid_token() {
    let auth = AuthConfig::bearer("secret123");
    assert!(!auth.validate("Bearer wrong"));
}

#[test]
fn test_bearer_auth_wrong_scheme() {
    let auth = AuthConfig::bearer("secret123");
    assert!(!auth.validate("Basic secret123"));
}

#[test]
fn test_bearer_auth_missing_prefix() {
    let auth = AuthConfig::bearer("secret123");
    assert!(!auth.validate("secret123"));
}

#[test]
fn test_no_auth_allows_anything() {
    let auth = AuthConfig::None;
    assert!(auth.validate("anything"));
    assert!(auth.validate(""));
    assert!(auth.validate("Bearer xyz"));
}

#[test]
fn test_basic_auth_constructor() {
    let auth = AuthConfig::basic("user", "pass");
    // Constructor should succeed
    assert!(matches!(auth, AuthConfig::Basic { .. }));
}

// ============= Edge Cases =============

#[test]
fn test_filter_default_allows_all() {
    let filter = IpFilter::default();
    assert!(filter.is_allowed("127.0.0.1".parse().unwrap()));
    assert!(filter.is_allowed("192.168.1.1".parse().unwrap()));
}

#[test]
fn test_multiple_cidrs_overlapping() {
    let mut filter = IpFilter::new();
    filter.allow("192.168.0.0/16").unwrap();
    filter.allow("192.168.1.0/24").unwrap();

    assert!(filter.is_allowed("192.168.1.1".parse().unwrap()));
    assert!(filter.is_allowed("192.168.2.1".parse().unwrap()));
}

#[test]
fn test_mixed_ipv4_and_ipv6() {
    let mut filter = IpFilter::new();
    filter.allow("127.0.0.1").unwrap();
    filter.allow("::1").unwrap();

    // IPv4
    assert!(filter.is_allowed("127.0.0.1".parse().unwrap()));

    // IPv6
    assert!(filter.is_allowed("::1".parse().unwrap()));

    // Wrong type
    assert!(!filter.is_allowed("127.0.0.2".parse().unwrap()));
    assert!(!filter.is_allowed("::2".parse().unwrap()));
}
