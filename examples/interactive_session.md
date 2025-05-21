# CloudLLM Interactive Session Example

This interactive Rust example demonstrates how to run an ongoing chat session with an OpenAI model (default: **GPT‑4.1 Nano**) using the CloudLLM API. It reads multi‑line user input, sends it to the model, shows a loading animation while waiting for the response, and prints the assistant’s reply in real time.

## What does this example do?

* **Reads** your OpenAI API key from the `OPEN_AI_SECRET` environment variable.
* **Instantiates** an `OpenAIClient` with a chosen model (by default `GPT4.1 Nano`).
* **Creates** an `LLMSession` with a system prompt (e.g., bilingual crypto/software journalist).
* **Enters** an infinite loop where it:

    1. Prompts for **multi‑line** user input, ending input when you type `\end` on its own line.
    2. Prints `Sending message...` and **spawns** a background task to display loading dots.
    3. Calls `session.send_message(...)` to get the assistant’s reply.
    4. Stops the loading animation and prints:

       ```
       Assistant:
       <model’s response>
       ```
* Repeats until you terminate the program.

## Key Components

* **`OpenAIClient`**: Wraps the OpenAI API; configured via your API key and selected model.
* **`LLMSession`**: Manages conversation history, context trimming, and sends messages through the client.
* **Loading spinner**: Utilizes `tokio::sync::watch` and an async task (`display_waiting_dots`) to show progress dots.

## How it works under the hood

1. **Setup**

    * Reads `OPEN_AI_SECRET` for the API key.
    * Creates `OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano)`.
    * Builds `LLMSession` with `Arc::new(client)` and a system prompt.
2. **Input loop**

    * Reads lines from `stdin` until a line equal to `\end` is entered.
    * Joins lines into a single user message.
3. **Request/Response**

    * Prints `Sending message...` and launches `display_waiting_dots` to animate.
    * Calls `session.send_message(Role::User, user_input).await`.
    * Upon completion, signals the spinner to stop and prints the assistant’s response.
4. **Repeat** for the next prompt.

## Prerequisites

* **Rust toolchain** (with `cargo`).
* **Tokio runtime**: Ensure you have `tokio = { version = "*", features = ["full"] }` in `Cargo.toml`.
* **Environment variable**: Set `OPEN_AI_SECRET` to your OpenAI API key:

  ```bash
  export OPEN_AI_SECRET=your-openai-key-here
  ```

## Running the example

From the root of the repository, run:

```bash
OPEN_AI_SECRET=your-openai-key-here cargo run --example interactive_session
```

Enjoy an interactive chat with your configured OpenAI model directly from the terminal! Feel free to customize the system prompt or model selection as needed.
