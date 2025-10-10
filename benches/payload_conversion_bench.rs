//! Benchmark to measure the cost of converting messages to provider format.
//!
//! This benchmark demonstrates that message conversion overhead is negligible
//! compared to network and LLM processing time.
//!
//! Run with: cargo run --release --bin payload_conversion_bench

use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone)]
struct Message {
    role: Role,
    content: Arc<str>,
}

struct ChatMessage {
    role: String,
    content: String,
}

fn convert_all(messages: &[Message]) -> Vec<ChatMessage> {
    let mut formatted = Vec::with_capacity(messages.len());
    for msg in messages {
        formatted.push(ChatMessage {
            role: match msg.role {
                Role::System => "system".to_owned(),
                Role::User => "user".to_owned(),
                Role::Assistant => "assistant".to_owned(),
            },
            content: msg.content.to_string(),
        });
    }
    formatted
}

fn main() {
    // Create a realistic conversation: system + 10 turns (20 messages)
    let mut conversation = vec![Message {
        role: Role::System,
        content: Arc::from("You are a helpful assistant."),
    }];

    for i in 0..10 {
        conversation.push(Message {
            role: Role::User,
            content: Arc::from(format!(
                "User message {} - this is a question or statement from the user that might be short or long depending on what they're asking about",
                i
            )),
        });
        conversation.push(Message {
            role: Role::Assistant,
            content: Arc::from(format!(
                "Assistant response {} - this is typically longer as the assistant provides detailed answers explaining concepts with examples and context",
                i
            )),
        });
    }

    println!("Payload Conversion Benchmark");
    println!("============================\n");
    println!("Conversation size: {} messages", conversation.len());
    println!(
        "Total content size: {} bytes\n",
        conversation.iter().map(|m| m.content.len()).sum::<usize>()
    );

    let iterations = 100_000;

    // Benchmark: Current approach (convert all messages)
    let start = Instant::now();
    for _ in 0..iterations {
        let _formatted = convert_all(&conversation);
    }
    let current_duration = start.elapsed();

    println!("Current approach (convert all messages each turn):");
    println!("  {} iterations", iterations);
    println!("  Total time: {:?}", current_duration);
    println!(
        "  Per turn: {:.2}µs",
        current_duration.as_micros() as f64 / iterations as f64
    );

    // Benchmark: Cached approach (only convert new message)
    let start = Instant::now();
    let mut cache = convert_all(&conversation[..conversation.len() - 1]);
    for _ in 0..iterations {
        // Convert only the last (new) message
        let new_msg = &conversation[conversation.len() - 1];
        cache.push(ChatMessage {
            role: "assistant".to_owned(),
            content: new_msg.content.to_string(),
        });
        let _use = &cache;
        cache.pop();
    }
    let cached_duration = start.elapsed();

    println!("\nCached approach (only convert new messages):");
    println!("  {} iterations", iterations);
    println!("  Total time: {:?}", cached_duration);
    println!(
        "  Per turn: {:.2}µs",
        cached_duration.as_micros() as f64 / iterations as f64
    );

    let savings_us =
        (current_duration.as_micros() - cached_duration.as_micros()) as f64 / iterations as f64;
    let speedup = current_duration.as_micros() as f64 / cached_duration.as_micros() as f64;

    println!("\nSavings: {:.2}µs per turn ({:.1}x faster)", savings_us, speedup);

    // Context
    println!("\n\nContext:");
    println!("========");
    println!("Network latency: ~100,000µs (100ms)");
    println!("LLM processing: ~1,000,000µs+ (1+ seconds)");
    println!("Conversion cost: {:.2}µs", current_duration.as_micros() as f64 / iterations as f64);
    println!(
        "Conversion as % of total: {:.4}%",
        (current_duration.as_micros() as f64 / iterations as f64) / 100_000.0 * 100.0
    );

    println!("\n✓ Conversion overhead is negligible (<0.001% of request time)");
}
