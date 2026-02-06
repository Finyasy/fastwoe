# Rust Best Practices for FastWOE (PyO3)

## Goal
Adopt a Rust-first workflow for the upcoming `fastwoe` core, with Python only as a binding/packaging interface.

## Tooling Baseline
- Install Rust via `rustup` and pin a stable toolchain.
- Use `cargo` for build/test/lint/format.
- Use `maturin` for Python extension build and wheel publishing.
- Avoid relying on `uv` for core build logic.

## Project Layout
- `crates/fastwoe-core`: pure Rust algorithms and statistics.
- `crates/fastwoe-py`: PyO3 binding crate exposing Python API.
- Keep Python-facing marshaling in bindings; keep business logic in core crate.

## Code Quality Standards
- Enforce formatting with `cargo fmt --all`.
- Enforce lints with `cargo clippy --all-targets --all-features -D warnings`.
- Run tests with `cargo test --all-features`.
- Prefer explicit error types (`thiserror`) and `Result<T, E>` propagation.
- Keep functions small, deterministic, and side-effect-light.

## Numerical and Statistical Safety
- Use deterministic ordering for category maps and output tables.
- Document and test numerical tolerance for parity with Python baseline.
- Avoid silent coercions for NaN/null/unknown categories; validate and fail clearly.
- Add regression fixtures for binary and multiclass WOE parity checks.

## PyO3 Interface Guidelines
- Minimize Python/Rust boundary crossings by batch processing arrays/tables.
- Use clear exception mapping (`ValueError` for bad inputs, `RuntimeError` for internal failures).
- Keep Python method names and behavior compatible with existing FastWOE API where possible.

## Performance Practices
- Benchmark core transforms and inference paths using representative datasets.
- Prefer contiguous memory patterns and avoid unnecessary allocations/clones.
- Profile before optimizing and document performance goals per release.

## Packaging and Release
- Build wheels with `maturin build` (Linux/macOS/Windows).
- Add CI for `fmt`, `clippy`, `test`, and wheel smoke tests.
- Publish pre-releases first; include migration notes for API behavior changes.

## Recommended Local Command Set
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -D warnings
cargo test --all-features
maturin develop
maturin build
```
