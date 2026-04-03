help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests (std + no_std)
	@cargo test --all-features
	@cargo test --no-default-features

clippy: ## Run clippy
	@cargo clippy --features rkyv-impl,rkyv/size_32,serde,nightly -- -D warnings

cq: ## Run code quality checks (formatting + clippy)
	@$(MAKE) fmt CHECK=1
	@$(MAKE) clippy

fmt: ## Format code
	@rustup component add --toolchain nightly rustfmt 2>/dev/null || true
	@cargo +nightly fmt --all $(if $(CHECK),-- --check,)

check: ## Type-check
	@cargo check --all-features --features rkyv/size_32

doc: ## Generate docs
	@cargo doc --no-deps

clean: ## Clean build artifacts
	@cargo clean

test-no-std: ## Run tests without std
	@cargo test --no-default-features

no-std: ## Verify no_std + WASM compatibility
	@rustup target add wasm32-unknown-unknown 2>/dev/null || true
	@cargo build --release --no-default-features --features serde --target wasm32-unknown-unknown

.PHONY: help test clippy cq fmt check doc clean test-no-std no-std
