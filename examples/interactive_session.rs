use std::env;
use std::io::{self, Write};

use cloudllm::cloudllm::client_wrapper::Role;
use cloudllm::cloudllm::clients::openai::OpenAIClient;
use cloudllm::cloudllm::llm_session::LLMSession;

#[tokio::main]
async fn main() {
    // Read OPENAI_AI_SECRET from environment variable
    let secret_key = env::var("OPEN_AI_SECRET")
        .expect("Please set the OPEN_AI_SECRET environment variable!");

    // Instantiate the OpenAI client
    let client = OpenAIClient::new(&secret_key, "gpt-4");

    // Set up the LLMSession
    let system_prompt = "You are an award winning bitcoin/blockchain/crypto/tech/software journalist for DiarioBitcoin, you are spanish/english bilingual, you can write in spanish at a professional journalist level, as well as a software engineer. You are hold a doctorate in economy and cryptography. When you answer you don't make any mentions of your credentials unless specifically asked about them.".to_string();
    let mut session = LLMSession::new(client, system_prompt);

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).expect("Failed to read line");
        let user_input = user_input.trim();

        // Send the user's message and get a response
        print!("Sending message...");
        let response = session.send_message(Role::User, user_input.to_string()).await.unwrap();

        // Print the assistant's response
        println!("Assistant: {}", response.content);
    }
}
