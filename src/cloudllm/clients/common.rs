use crate::client_wrapper::{MessageChunk, SendError, TokenUsage};
use futures_util::{Stream, StreamExt};
use openai_rust::chat;
use openai_rust2 as openai_rust;
use std::error::Error;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

/// Send a chat request, record its usage, and return the assistant’s content.
pub async fn send_and_track(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
) -> Result<String, Box<dyn Error>> {
    let mut chat_arguments = chat::ChatArguments::new(model, formatted_msgs);

    if let Some(search_params) = optional_search_parameters {
        chat_arguments = chat_arguments.with_search_parameters(search_params);
    }

    let response = api.create_chat(chat_arguments, url_path).await;

    match response {
        Ok(response) => {
            let usage = TokenUsage {
                input_tokens: response.usage.prompt_tokens as usize,
                output_tokens: response.usage.completion_tokens as usize,
                total_tokens: response.usage.total_tokens as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().unwrap() = Some(usage);

            // Return the assistant’s content
            Ok(response.choices[0].message.content.clone())
        }
        Err(err) => {
            log::error!(
                "cloudllm::clients::common::send_and_track(...): OpenAI API Error: {}",
                err
            ); // Log the entire error
            Err(err.into()) // Convert the error to Box<dyn Error>
        }
    }
}

/// Send a streaming chat request and return a stream of message chunks.
/// Note: Token usage tracking is not available for streaming responses.
pub async fn send_and_track_stream(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
) -> Result<Pin<Box<dyn Stream<Item = Result<MessageChunk, SendError>>>>, Box<dyn Error>> {
    let mut chat_arguments = chat::ChatArguments::new(model, formatted_msgs);

    if let Some(search_params) = optional_search_parameters {
        chat_arguments = chat_arguments.with_search_parameters(search_params);
    }

    let chunk_stream = api.create_chat_stream(chat_arguments, url_path).await?;

    // Map the chunks to our MessageChunk type
    let message_stream = chunk_stream.map(|chunk_result| {
        match chunk_result {
            Ok(chunk) => {
                let content = chunk.choices[0]
                    .delta
                    .content
                    .clone()
                    .unwrap_or_default();
                let is_final = chunk.choices[0].finish_reason.is_some();
                
                Ok(MessageChunk { content, is_final })
            }
            Err(err) => {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Stream error: {}", err)
                )) as SendError)
            }
        }
    });

    Ok(Box::pin(message_stream))
}
