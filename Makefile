.PHONY: build check test clippy clean release

build:
	cargo build

check:
	cargo check

test:
	cargo fmt --check
	cargo test --workspace
	cargo clippy --workspace -- -D warnings

clippy:
	cargo clippy --workspace -- -D warnings

clippy-all:
	cargo clippy --all-targets --all-features -- -D warnings

fmt:
	cargo fmt --all

clean:
	cargo clean

# Override with VERSION=patch, VERSION=major, or VERSION=X.Y.Z.
VERSION ?= patch

release:
	cargo release version $(VERSION) --workspace --execute --no-confirm
