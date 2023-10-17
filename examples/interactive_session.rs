use std::env;
use std::io::{self, Write};

use tokio::sync::watch;
use tokio::time::{Duration, sleep};

use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::LLMSession;

// Run from the root folder of the repo as follows:
// OPEN_AI_SECRET=your-open-ai-key-here cargo run --example interactive_session

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
        print!("\n\nYou [type '\\end' in a separate line to submit prompt]:\n");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        loop {
            let mut line = String::new();
            io::stdin().read_line(&mut line).expect("Failed to read line");

            // Check for the end sequence in the line
            if line.trim() == "\\end" {
                break;
            } else {
                user_input.push_str(&line); // Keep the raw line, preserving newlines
            }
        }

        if user_input.is_empty() {
            println!("Input is empty. Try again.");
            continue;
        }

        // Send the user's message and get a response
        println!("Sending message...");
        let (tx, rx) = watch::channel(true);
        tokio::spawn(display_waiting_dots(rx, 3));

        let response = session.send_message(Role::User, user_input.to_string()).await.unwrap();
        tx.send(false).unwrap();

        // Print the assistant's response
        println!("\nAssistant:\n{}\n", response.content);
    }
}

async fn display_waiting_dots(rx: watch::Receiver<bool>, num_dots: usize) {
    let mut loading = true;
    while loading {
        for _ in 0..num_dots {
            if !rx.borrow().clone() {
                break;
            }
            print!(".");
            io::stdout().flush().unwrap();
            sleep(Duration::from_millis(500)).await;
        }
        print!("\r");
        for _ in 0..num_dots {
            print!(" ");
        }
        print!("\r");
        io::stdout().flush().unwrap();
        loading = rx.borrow().clone();
    }
    print!("\r");
    for _ in 0..num_dots {
        print!(" ");
    }
    print!("\r\n");
    io::stdout().flush().unwrap();
}
