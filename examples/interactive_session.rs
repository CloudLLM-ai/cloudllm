use std::env;
use std::io::{self, Write};

use tokio::sync::watch;
use tokio::time::{sleep, Duration};

use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::LLMSession;
// Run from the root folder of the repo as follows:
// OPEN_AI_SECRET=your-open-ai-key-here cargo run --example interactive_session

#[tokio::main]
async fn main() {
    // Read OPENAI_AI_SECRET from environment variable
    let secret_key =
         env::var("OPEN_AI_SECRET").expect("Please set the OPEN_AI_SECRET environment variable!");

    // Read GEMINI_API_KEY from environment variable
    //let secret_key =
    //    env::var("GEMINI_API_KEY").expect("Please set the GEMINI_API_KEY environment variable!");

    // Read the XAI_API_KEY from environment variable
    //let secret_key =
    //    env::var("XAI_API_KEY").expect("Please set the XAI_API_KEY environment variable!");

    // Instantiate the OpenAI client
    let client = OpenAIClient::new_with_model_enum(&secret_key, cloudllm::clients::openai::Model::GPT5Nano);
    //let client = OpenAIClient::new_with_model_string(&secret_key, "gpt-5-nano"); // hardcode the string

    // Instantiate the Gemini client
    //let client = cloudllm::clients::gemini::GeminiClient::new_with_model_enum(&secret_key, cloudllm::clients::gemini::Model::Gemini25Flash);

    // Instantiate the Grok client
    //let client = GrokClient::new_with_model_enum(&secret_key, clients::grok::Model::Grok3MiniBeta);

    // Set up the LLMSession
    let system_prompt = "You are an award winning bitcoin/blockchain/crypto/tech/software journalist for DiarioBitcoin, you are spanish/english bilingual, you can write in spanish at a professional journalist level, as well as a software engineer. You are hold a doctorate in economy and cryptography. When you answer you don't make any mentions of your credentials unless specifically asked about them.".to_string();
    let max_tokens = 1024; // Set a small context window for testing conversation history pruning
    let mut session = LLMSession::new(std::sync::Arc::new(client), system_prompt, max_tokens);

    loop {
        print!("\n\nYou [type '\\end' in a separate line to submit prompt]:\n");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        loop {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("Failed to read line");

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

        let response_result = session
            .send_message(Role::User, user_input.to_string(), None)
            .await;

        tx.send(false).unwrap();

        let response = match response_result {
            Ok(r) => {
                r
            }
            Err(err ) => {
                println!("\n\n[Error sending message:] {}\n", err);
                continue; // Skip to the next iteration of the loop
            }
        };

        let token_usage = session.token_usage();

        // Print the assistant's response
        println!(
            "\nToken Usage: <input tokens:{}, output tokens:{}, total tokens:{}, max tokens: {}>\nAssistant:\n{}\n",
            token_usage.input_tokens,
            token_usage.output_tokens,
            token_usage.total_tokens,
            session.get_max_tokens(),
            response.content
        );
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
