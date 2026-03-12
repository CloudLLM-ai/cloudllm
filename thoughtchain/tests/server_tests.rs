#![cfg(feature = "server")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use serde_json::json;
use thoughtchain::server::{
    mcp_router, rest_router, standard_mcp_router, ThoughtChainServerConfig,
    ThoughtChainServiceConfig,
};
use thoughtchain::StorageAdapterKind;
use tower::util::ServiceExt;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn unique_chain_dir() -> PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "thoughtchain_server_test_{}_{}",
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn env_mutex() -> &'static Mutex<()> {
    ENV_MUTEX.get_or_init(|| Mutex::new(()))
}

#[test]
fn server_config_parses_thoughtchain_verbose_env_values() {
    let _guard = env_mutex().lock().unwrap();
    let original = std::env::var("THOUGHTCHAIN_VERBOSE").ok();

    for (raw_value, expected) in [
        ("1", true),
        ("0", false),
        ("true", true),
        ("false", false),
        ("TRUE", true),
        ("FALSE", false),
        ("unexpected", false),
    ] {
        std::env::set_var("THOUGHTCHAIN_VERBOSE", raw_value);
        let config = ThoughtChainServerConfig::from_env();
        assert_eq!(
            config.service.verbose, expected,
            "raw value {raw_value:?} should parse to {expected}"
        );
    }

    std::env::remove_var("THOUGHTCHAIN_VERBOSE");
    assert!(!ThoughtChainServerConfig::from_env().service.verbose);

    if let Some(original) = original {
        std::env::set_var("THOUGHTCHAIN_VERBOSE", original);
    } else {
        std::env::remove_var("THOUGHTCHAIN_VERBOSE");
    }
}

#[tokio::test]
async fn mcp_router_lists_thoughtchain_tools() {
    let dir = unique_chain_dir();
    let router = mcp_router(ThoughtChainServiceConfig::new(
        dir.clone(),
        "server-test",
        StorageAdapterKind::Jsonl,
    ));

    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tools/list")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let tools = json["tools"].as_array().unwrap();
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_append"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_append_retrospective"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_list_chains"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_list_agents"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_get_agent"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_list_agent_registry"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_upsert_agent"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_set_agent_description"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_add_agent_alias"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_add_agent_key"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_revoke_agent_key"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_disable_agent"));
    assert!(tools.iter().any(|tool| tool["name"] == "thoughtchain_head"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn rest_router_bootstraps_and_reports_head() {
    let dir = unique_chain_dir();
    let router = rest_router(ThoughtChainServiceConfig::new(
        dir.clone(),
        "server-test",
        StorageAdapterKind::Jsonl,
    ));

    let health = router
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let bootstrap = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/bootstrap")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "server-test",
                        "storage_adapter": "binary",
                        "content": "Bootstrap memory for the server test.",
                        "importance": 1.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bootstrap.status(), StatusCode::OK);

    let head = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/head")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "server-test"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(head.status(), StatusCode::OK);
    let body = axum::body::to_bytes(head.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["thought_count"], 1);
    assert_eq!(json["integrity_ok"], true);

    let chains = router
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/chains")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chains.status(), StatusCode::OK);
    let chains_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(chains.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let summary = chains_json["chains"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["chain_key"] == "server-test")
        .unwrap();
    assert_eq!(summary["version"], 1);
    assert_eq!(summary["storage_adapter"], "binary");
    assert_eq!(summary["thought_count"], 1);
    assert_eq!(summary["agent_count"], 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn rest_router_supports_shared_chain_agent_identity() {
    let dir = unique_chain_dir();
    let router = rest_router(ThoughtChainServiceConfig::new(
        dir.clone(),
        "shared-chain",
        StorageAdapterKind::Jsonl,
    ));

    let append = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/thoughts")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "shared-chain",
                        "agent_id": "agent-42",
                        "agent_name": "Planner",
                        "agent_owner": "ops-team",
                        "thought_type": "Decision",
                        "content": "Retry with exponential backoff."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(append.status(), StatusCode::OK);

    let search = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "shared-chain",
                        "agent_names": ["Planner"],
                        "agent_owners": ["ops-team"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(search.status(), StatusCode::OK);
    let body = axum::body::to_bytes(search.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let thoughts = json["thoughts"].as_array().unwrap();
    assert_eq!(thoughts.len(), 1);
    assert_eq!(thoughts[0]["agent_id"], "agent-42");
    assert_eq!(thoughts[0]["agent_name"], "Planner");
    assert_eq!(thoughts[0]["agent_owner"], "ops-team");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn rest_router_appends_retrospective_with_defaults() {
    let dir = unique_chain_dir();
    let router = rest_router(ThoughtChainServiceConfig::new(
        dir.clone(),
        "shared-chain",
        StorageAdapterKind::Jsonl,
    ));

    let append = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/retrospectives")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "shared-chain",
                        "agent_id": "astro",
                        "agent_name": "Astro",
                        "content": "After a repeated tool-call failure, respond to every tool_call_id before sending the next model request."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(append.status(), StatusCode::OK);
    let body = axum::body::to_bytes(append.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["thought"]["thought_type"], "LessonLearned");
    assert_eq!(json["thought"]["role"], "Retrospective");
    assert_eq!(json["thought"]["agent_name"], "Astro");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn rest_router_lists_chains_and_agents() {
    let dir = unique_chain_dir();
    let router = rest_router(ThoughtChainServiceConfig::new(
        dir.clone(),
        "shared-brain",
        StorageAdapterKind::Jsonl,
    ));

    let append_one = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/thoughts")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "shared-brain",
                        "agent_id": "astro",
                        "agent_name": "Astro",
                        "thought_type": "Decision",
                        "content": "Use the shared chain for memory."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(append_one.status(), StatusCode::OK);

    let append_two = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/thoughts")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "shared-brain",
                        "agent_id": "apollo",
                        "agent_name": "Apollo",
                        "agent_owner": "@gubatron",
                        "thought_type": "Insight",
                        "content": "Shared memory helps future agents resume."
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(append_two.status(), StatusCode::OK);

    let chains = router
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/chains")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chains.status(), StatusCode::OK);
    let chains_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(chains.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let chain_keys = chains_json["chain_keys"].as_array().unwrap();
    assert!(chain_keys.iter().any(|value| value == "shared-brain"));
    assert_eq!(chains_json["default_chain_key"], "shared-brain");
    let summary = chains_json["chains"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["chain_key"] == "shared-brain")
        .unwrap();
    assert_eq!(summary["version"], 1);
    assert_eq!(summary["storage_adapter"], "jsonl");
    assert_eq!(summary["thought_count"], 2);
    assert_eq!(summary["agent_count"], 2);

    let agents = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agents")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "shared-brain"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(agents.status(), StatusCode::OK);
    let agents_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(agents.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let agent_entries = agents_json["agents"].as_array().unwrap();
    assert!(agent_entries
        .iter()
        .any(|agent| agent["agent_name"] == "Astro" && agent["agent_id"] == "astro"));
    assert!(agent_entries.iter().any(|agent| {
        agent["agent_name"] == "Apollo"
            && agent["agent_id"] == "apollo"
            && agent["agent_owner"] == "@gubatron"
    }));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn rest_router_manages_agent_registry_records() {
    let dir = unique_chain_dir();
    let router = rest_router(ThoughtChainServiceConfig::new(
        dir.clone(),
        "registry-admin",
        StorageAdapterKind::Binary,
    ));

    let upsert = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agents/upsert")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "registry-admin",
                        "agent_id": "agent-admin",
                        "display_name": "Registry Admin",
                        "agent_owner": "@gubatron",
                        "description": "Admin test agent",
                        "status": "active"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(upsert.status(), StatusCode::OK);

    let alias = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agents/aliases")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "registry-admin",
                        "agent_id": "agent-admin",
                        "alias": "astro-admin"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(alias.status(), StatusCode::OK);

    let add_key = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agents/keys")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "registry-admin",
                        "agent_id": "agent-admin",
                        "key_id": "main-ed25519",
                        "algorithm": "ed25519",
                        "public_key_bytes": [1, 2, 3, 4]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(add_key.status(), StatusCode::OK);

    let revoke_key = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agents/keys/revoke")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "registry-admin",
                        "agent_id": "agent-admin",
                        "key_id": "main-ed25519"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoke_key.status(), StatusCode::OK);

    let disable = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agents/disable")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "registry-admin",
                        "agent_id": "agent-admin"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(disable.status(), StatusCode::OK);

    let get_agent = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agent")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "registry-admin",
                        "agent_id": "agent-admin"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_agent.status(), StatusCode::OK);
    let agent_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(get_agent.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(agent_json["agent"]["display_name"], "Registry Admin");
    assert_eq!(agent_json["agent"]["owner"], "@gubatron");
    assert_eq!(agent_json["agent"]["description"], "Admin test agent");
    assert_eq!(agent_json["agent"]["status"], "Revoked");
    assert!(agent_json["agent"]["aliases"]
        .as_array()
        .unwrap()
        .iter()
        .any(|alias| alias == "astro-admin"));
    assert_eq!(agent_json["agent"]["public_keys"][0]["algorithm"], "Ed25519");
    assert!(agent_json["agent"]["public_keys"][0]["revoked_at"].is_string());

    let registry = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/agent-registry")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "chain_key": "registry-admin"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(registry.status(), StatusCode::OK);
    let registry_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(registry.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(registry_json["agents"].as_array().unwrap().len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn live_mcp_server_supports_standard_initialize_and_tools_list() {
    let dir = unique_chain_dir();
    let router = standard_mcp_router(ThoughtChainServiceConfig::new(
        dir.clone(),
        "server-test",
        StorageAdapterKind::Jsonl,
    ));
    let client_addr = std::net::SocketAddr::from(([127, 0, 0, 1], 49000));

    let mut initialize_request = Request::builder()
        .method("POST")
        .uri("/")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "thoughtchain-test",
                        "version": "0.1.0"
                    }
                }
            })
            .to_string(),
        ))
        .unwrap();
    initialize_request
        .extensions_mut()
        .insert(ConnectInfo(client_addr));
    let initialize = router.clone().oneshot(initialize_request).await.unwrap();
    assert_eq!(initialize.status(), StatusCode::OK);
    assert_eq!(
        initialize
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let initialize_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(initialize.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(initialize_json["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(
        initialize_json["result"]["serverInfo"]["name"],
        "thoughtchain"
    );

    let mut initialized_request = Request::builder()
        .method("POST")
        .uri("/")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })
            .to_string(),
        ))
        .unwrap();
    initialized_request
        .extensions_mut()
        .insert(ConnectInfo(client_addr));
    let initialized = router.clone().oneshot(initialized_request).await.unwrap();
    assert_eq!(initialized.status(), StatusCode::ACCEPTED);

    let mut tools_list_request = Request::builder()
        .method("POST")
        .uri("/")
        .header("content-type", "application/json")
        .header("MCP-Protocol-Version", "2025-06-18")
        .body(Body::from(
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            })
            .to_string(),
        ))
        .unwrap();
    tools_list_request
        .extensions_mut()
        .insert(ConnectInfo(client_addr));
    let tools_list = router.oneshot(tools_list_request).await.unwrap();
    assert_eq!(tools_list.status(), StatusCode::OK);
    let tools_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(tools_list.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let tools = tools_json["result"]["tools"].as_array().unwrap();
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_append"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_append_retrospective"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_list_chains"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_list_agents"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_get_agent"));
    assert!(tools
        .iter()
        .any(|tool| tool["name"] == "thoughtchain_upsert_agent"));
    assert!(tools.iter().any(|tool| tool["name"] == "thoughtchain_head"));

    let _ = std::fs::remove_dir_all(&dir);
}
