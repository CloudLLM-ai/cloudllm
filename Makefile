# Makefile
.PHONY: build release clean fmt check test install doc help tasks clippy publish publish-dry-run build-thoughtchaind

default: help
CARGO_CMD=/usr/bin/env cargo

# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------------------------------------------------

# Default target (ensures formatting before building)
build: fmt build-thoughtchaind ## Build the full workspace in release mode (runs fmt first)
	${CARGO_CMD} build --workspace --release

# Explicit daemon build so the ThoughtChain binary is always validated too
build-thoughtchaind: ## Build the thoughtchaind binary in release mode
	${CARGO_CMD} build -p thoughtchain --features server --bin thoughtchaind --release

# Full release process (ensures everything runs in the correct order)
release: fmt check clippy build test doc ## Perform a full release (fmt, check, clippy, build, test, doc)

# Format the code
fmt: ## Format the code using cargo fmt
	${CARGO_CMD} fmt

# Check for errors without building
check: ## Run cargo check to analyze the code without compiling
	${CARGO_CMD} check --workspace
	${CARGO_CMD} check -p thoughtchain --features server --bin thoughtchaind

# Strict linter, fails on warning and suggests fixes
clippy: ## Run clippy across the workspace and fail on warnings
	${CARGO_CMD} fmt
	${CARGO_CMD} clippy --workspace --all-targets --all-features -- -D warnings

# Run tests
test: ## Run tests using cargo test
	${CARGO_CMD} test --workspace
	${CARGO_CMD} test -p thoughtchain --features server

# Generate documentation
doc: ## Generate project documentation using cargo doc
	${CARGO_CMD} doc --workspace --all-features

# Publish workspace crates to crates.io in dependency order
publish: ## Publish mcp, thoughtchain, then cloudllm to crates.io
	${CARGO_CMD} publish -p mcp
	${CARGO_CMD} publish -p thoughtchain
	${CARGO_CMD} publish -p cloudllm

# Dry-run workspace publish in dependency order
publish-dry-run: ## Dry-run publish for mcp, thoughtchain, then cloudllm
	${CARGO_CMD} publish -p mcp --dry-run
	${CARGO_CMD} publish -p thoughtchain --dry-run
	${CARGO_CMD} publish -p cloudllm --dry-run

# Clean build artifacts
clean: ## Remove build artifacts using cargo clean
	${CARGO_CMD} clean

# Show all available tasks
help tasks: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
