use std::sync::Mutex;
use openai_rust2 as openai_rust;
use openai_rust::chat;
use crate::client_wrapper::TokenUsage;
use std::error::Error;

/// Send a chat request, record its usage, and return the assistant’s content.
pub async fn send_and_track(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
) -> Result<String, Box<dyn Error>> {
    let response = api.create_chat(chat::ChatArguments::new(model, formatted_msgs), url_path).await;

    match response {
        Ok(response) => {
            // Log the response
            // Pull out the usage
            let usage = TokenUsage {
                input_tokens:  response.usage.prompt_tokens     as usize,
                output_tokens: response.usage.completion_tokens as usize,
                total_tokens:  response.usage.total_tokens      as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().unwrap() = Some(usage);

            // Return the assistant’s content
            Ok(response.choices[0].message.content.clone())
        }
        Err(err) => {
            log::error!("cloudllm::clients::common::send_and_track(...): OpenAI API Error: {}", err); // Log the entire error
            Err(err.into()) // Convert the error to Box<dyn Error>
        }
    }
}
