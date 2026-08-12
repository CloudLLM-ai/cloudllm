//! Isolated tests for streaming session bookkeeping (rollback / usage slot).

use async_trait::async_trait;
use cloudllm::client_wrapper::{
    Message, MessageChunk, MessageChunkStream, MessageStreamFuture, Role, TokenUsage,
    ToolDefinition,
};
use cloudllm::{ClientWrapper, LLMSession};
use std::sync::Arc;
use tokio::sync::Mutex;

struct StreamMock {
    usage: Mutex<Option<TokenUsage>>,
    mode: StreamMockMode,
}

#[derive(Clone, Copy)]
enum StreamMockMode {
    OpenFails,
    ReturnsNone,
    OpensThenErrors,
    OpensEmpty,
}

impl StreamMock {
    fn new(mode: StreamMockMode) -> Arc<Self> {
        Arc::new(Self {
            usage: Mutex::new(None),
            mode,
        })
    }
}

#[async_trait]
impl ClientWrapper for StreamMock {
    async fn send_message(
        &self,
        _messages: &[Message],
        _tools: Option<&[ToolDefinition]>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        *self.usage.lock().await = Some(TokenUsage {
            input_tokens: 3,
            output_tokens: 2,
            total_tokens: 5,
        });
        Ok(Message {
            role: Role::Assistant,
            content: Arc::from("blocked"),
            tool_calls: vec![],
        })
    }

    fn send_message_stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&[ToolDefinition]>,
    ) -> MessageStreamFuture<'a> {
        let mode = self.mode;
        Box::pin(async move {
            match mode {
                StreamMockMode::OpenFails => Err(Box::new(std::io::Error::other("open failed"))
                    as Box<dyn std::error::Error + Send + Sync>),
                StreamMockMode::ReturnsNone => Ok(None),
                StreamMockMode::OpensThenErrors => {
                    let s = futures_util::stream::iter(vec![
                        Ok(MessageChunk {
                            content: "hi".into(),
                            ..MessageChunk::default()
                        }),
                        Err(Box::new(std::io::Error::other("mid-stream"))
                            as Box<dyn std::error::Error + Send + Sync>),
                    ]);
                    Ok(Some(Box::pin(s) as MessageChunkStream))
                }
                StreamMockMode::OpensEmpty => {
                    let s = futures_util::stream::empty();
                    Ok(Some(Box::pin(s) as MessageChunkStream))
                }
            }
        })
    }

    fn model_name(&self) -> &str {
        "mock-stream"
    }

    fn provider_name(&self) -> &str {
        "mock"
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.usage)
    }
}

#[tokio::test]
async fn send_message_stream_rolls_back_on_open_error() {
    let client = StreamMock::new(StreamMockMode::OpenFails);
    let mut session = LLMSession::new(client, "sys".into(), 8_192);
    let err = session
        .send_message_stream(Role::User, "hello".into(), None)
        .await;
    assert!(err.is_err());
    assert!(session.get_conversation_history().is_empty());
}

#[tokio::test]
async fn send_message_stream_rolls_back_on_none() {
    let client = StreamMock::new(StreamMockMode::ReturnsNone);
    let mut session = LLMSession::new(client, "sys".into(), 8_192);
    let result = session
        .send_message_stream(Role::User, "hello".into(), None)
        .await
        .unwrap();
    assert!(result.is_none());
    assert!(session.get_conversation_history().is_empty());
}

#[tokio::test]
async fn send_message_stream_keeps_user_when_stream_opens() {
    let client = StreamMock::new(StreamMockMode::OpensThenErrors);
    let mut session = LLMSession::new(client, "sys".into(), 8_192);
    let stream = session
        .send_message_stream(Role::User, "hello".into(), None)
        .await
        .unwrap();
    assert!(stream.is_some());
    assert_eq!(session.get_conversation_history().len(), 1);
    session.rollback_last_message();
    assert!(session.get_conversation_history().is_empty());
}

#[tokio::test]
async fn commit_streamed_reply_writes_usage_slot() {
    let client = StreamMock::new(StreamMockMode::OpensEmpty);
    let mut session = LLMSession::new(client.clone(), "sys".into(), 8_192);
    session.inject_message(Role::User, "hello".into());
    session
        .commit_streamed_reply(
            Message {
                role: Role::Assistant,
                content: Arc::from("hi"),
                tool_calls: vec![],
            },
            Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 4,
                total_tokens: 14,
            }),
        )
        .await;
    let usage = session.last_token_usage().await.unwrap();
    assert_eq!(usage.total_tokens, 14);
    assert_eq!(session.token_usage().total_tokens, 14);
    assert_eq!(session.get_conversation_history().len(), 2);
}
