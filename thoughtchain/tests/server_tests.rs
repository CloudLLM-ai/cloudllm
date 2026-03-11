#![cfg(feature = "server")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use serde_json::json;
use thoughtchain::server::{
    mcp_router, rest_router, standard_mcp_router, ThoughtChainServiceConfig,
};
use thoughtchain::StorageAdapterKind;
use tower::util::ServiceExt;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    assert!(tools.iter().any(|tool| tool["name"] == "thoughtchain_head"));

    let _ = std::fs::remove_dir_all(&dir);
}
