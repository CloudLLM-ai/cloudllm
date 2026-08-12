//! Isolated tests for the MentisDB-backed `memory` tool.

use cloudllm::live_console::LiveConsoleHandler;
use cloudllm::tool_protocol::ToolProtocol;
use cloudllm::tool_protocols::MentisDbMemoryProtocol;

#[tokio::test]
async fn put_get_list_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let handle = LiveConsoleHandler::open_embedded_mentisdb_at(
        tmp.path().to_path_buf(),
        "kv-test".into(),
    )
    .unwrap();
    let proto = MentisDbMemoryProtocol::new(handle.db.clone(), "writer");
    proto
        .put_value("current_game_html", "<html>ok</html>")
        .await
        .unwrap();
    assert_eq!(
        proto.get_value("current_game_html").as_deref(),
        Some("<html>ok</html>")
    );
    assert_eq!(proto.list_keys(), vec!["current_game_html".to_string()]);

    let result = proto
        .execute(
            "memory",
            serde_json::json!({"command": "G", "key": "current_game_html"}),
        )
        .await
        .unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn hydrate_latest_after_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    {
        let handle =
            LiveConsoleHandler::open_embedded_mentisdb_at(dir.clone(), "kv-test".into()).unwrap();
        let proto = MentisDbMemoryProtocol::new(handle.db.clone(), "writer");
        proto.put_value("k", "v1").await.unwrap();
        proto.put_value("k", "v2").await.unwrap();
    }
    let handle = LiveConsoleHandler::open_embedded_mentisdb_at(dir, "kv-test".into()).unwrap();
    let proto = MentisDbMemoryProtocol::new(handle.db.clone(), "writer");
    assert_eq!(proto.get_value("k").as_deref(), Some("v2"));
}
