![image](https://github.com/CloudLLM-ai/cloudllm/assets/163977/4deabd8c-767e-46de-bb36-17123dd8d6a7)

# CloudLLM

CloudLLM is a Rust library designed to seamlessly bridge applications with remote Language Learning Models (LLMs) across various platforms. With CloudLLM, you can integrate pay-as-you-go LLM APIs like OpenAI's and more, all under one unified abstraction for your app.

CloudLLM is initially implemented for Rust, but we'll provide wrappers for most popular programming languages in the near future.

## Features

- **Unified Interface**: Interact with multiple LLMs using a single, consistent API.
- **Pay-as-you-go Integration**: Designed to work efficiently with pay-as-you-go LLM platforms.
- **Extendable**: Easily add new LLM platform clients as they emerge.
- **Asynchronous Support**: Built with async operations for non-blocking calls.

## Quick Start

```rust
// Example code on setting up a session and communicating with an LLM (this is just a placeholder for now).
```

## Installation

Add CloudLLM to your `Cargo.toml`:

```toml
[dependencies]
cloudllm = "0.1.0" # Use the latest version
```

## Supported LLM Platforms

- OpenAI
- Claude (Coming Soon)
- AWS Bedrock (Coming Soon)
- ... and more to come!

## Usage

Refer to the `examples/` directory to see how you can set up sessions and interact with various LLM platforms using CloudLLM.

## Contributing

Contributions to CloudLLM are always welcome! Whether it's feature suggestions, bug reporting, or code improvements, all contributions are appreciated.

If you are to send a pull request, please make a separate branch out of `main`. Try to minimize the scope of your contribution to one issue per pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

**Angel Leon**

---

[CloudLLM.ai](https://cloudllm.ai)
