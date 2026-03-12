#[path = "thoughtchaind.rs"]
mod thoughtchaind;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    thoughtchaind::run().await
}
