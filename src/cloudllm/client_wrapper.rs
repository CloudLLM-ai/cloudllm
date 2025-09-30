use async_trait::async_trait;
use openai_rust2 as openai_rust;
/// A ClientWrapper is a wrapper around a specific cloud LLM service.
/// It provides a common interface to interact with the LLMs.
/// It does not keep track of the conversation/session, for that we use an LLMSession
/// which keeps track of the conversation history and other session-specific data
/// and uses a ClientWrapper to interact with the LLM.
// src/client_wrapper
use std::error::Error;
use tokio::sync::Mutex;

/// Represents the possible roles for a message.
#[derive(Clone)]
pub enum Role {
    System,
    // set by the developer to steer the model's responses
    User,
    // a message sent by a human user (or app user)
    Assistant, // lets the model know the content was generated as a response to a user message
               // Add other roles as needed
}

/// How many tokens were spent on prompt vs. completion.
#[derive(Clone, Debug)]
pub struct TokenUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

/// Represents a generic message to be sent to an LLM.
#[derive(Clone)]
pub struct Message {
    /// The role associated with the message.
    pub role: Role,
    /// The actual content of the message.
    pub content: String,
}

/// Trait defining the interface to interact with various LLM services.
#[async_trait]
pub trait ClientWrapper: Send + Sync {
    /// Send a message to the LLM and get a response.
    /// - `messages`: The messages to send in the request.
    async fn send_message(
        &self,
        messages: Vec<Message>,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>>;

    /// Hook to retrieve usage from the *last* send_message() call.
    /// Default impl returns None so existing wrappers don’t break.
    async fn get_last_usage(&self) -> Option<TokenUsage> {
        if let Some(slot) = self.usage_slot() {
            slot.lock().await.clone()
        } else {
            None
        }
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        // ClientWrapper implementations supporting TokenUsage tracking should return a Mutex<Option<TokenUsage>> by overriding this method.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_mutex_usage_tracking() {
        // Test that we can lock and update the mutex in an async context
        let usage_mutex = Mutex::new(Some(TokenUsage {
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
        }));

        // Lock and update the value
        {
            let mut guard = usage_mutex.lock().await;
            *guard = Some(TokenUsage {
                input_tokens: 100,
                output_tokens: 200,
                total_tokens: 300,
            });
        }

        // Read the value back
        let guard = usage_mutex.lock().await;
        let usage = guard.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 200);
        assert_eq!(usage.total_tokens, 300);
    }

    #[tokio::test]
    async fn test_concurrent_mutex_access() {
        use std::sync::Arc;

        // Test that multiple async tasks can access the mutex concurrently
        let usage_mutex = Arc::new(Mutex::new(Some(TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
        })));

        let mut handles = vec![];

        // Spawn 10 tasks that all update the mutex
        for i in 0..10 {
            let mutex_clone = Arc::clone(&usage_mutex);
            let handle = tokio::spawn(async move {
                let mut guard = mutex_clone.lock().await;
                if let Some(ref mut usage) = *guard {
                    usage.input_tokens += i;
                    usage.total_tokens += i;
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify the final value
        let guard = usage_mutex.lock().await;
        let usage = guard.as_ref().unwrap();
        // Sum of 0..10 is 45
        assert_eq!(usage.input_tokens, 45);
        assert_eq!(usage.total_tokens, 45);
    }
}
