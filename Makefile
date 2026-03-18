help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests (std + no_std)
	@cargo test --all-features
	@cargo test --no-default-features

clippy: ## Run clippy
	@cargo clippy --all-features -- -D warnings

fmt: ## Format code
	@cargo +nightly fmt --all

check: ## Type-check
	@cargo check --all-features

doc: ## Generate docs
	@cargo doc --no-deps

clean: ## Clean build artifacts
	@cargo clean

test-no-std: ## Run tests without std
	@cargo test --no-default-features

no-std: ## Verify no_std + WASM compatibility
	@rustup target add wasm32-unknown-unknown 2>/dev/null || true
	@cargo build --release --no-default-features --features serde --target wasm32-unknown-unknown

.PHONY: help test clippy fmt check doc clean test-no-std no-std
