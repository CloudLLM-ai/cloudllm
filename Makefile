# Makefile
.PHONY: build release clean fmt check test install doc help tasks clippy publish

default: help
CARGO_CMD=/usr/bin/env cargo

# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------------------------------------------------

# Default target (ensures formatting before building)
build: fmt ## Build the project in release mode (runs fmt first)
	${CARGO_CMD} build --release

# Full release process (ensures everything runs in the correct order)
release: fmt check clippy build test doc ## Perform a full release (fmt, check, clippy, build, test, doc)

# Format the code
fmt: ## Format the code using cargo fmt
	${CARGO_CMD} fmt

# Check for errors without building
check: ## Run cargo check to analyze the code without compiling
	${CARGO_CMD} check

# Strict linter, fails on warning and suggests fixes
clippy: ## Checks a package to catch common mistakes and improve your Rust code
	${CARGO_CMD} fmt
	${CARGO_CMD} clippy --package cloudllm --lib
	${CARGO_CMD} clippy -- -D warnings

# Run tests
test: ## Run tests using cargo test
	${CARGO_CMD} test

# Generate documentation
doc: ## Generate project documentation using cargo doc
	${CARGO_CMD} doc

# Publish to crates.io
publish: ## Publish the crate to crates.io
	${CARGO_CMD} publish

# Clean build artifacts
clean: ## Remove build artifacts using cargo clean
	${CARGO_CMD} clean

# Show all available tasks
help tasks: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
