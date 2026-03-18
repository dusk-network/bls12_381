# AGENTS.md — dusk-bls12_381

## Care Level: Cryptographic — Elevated

This is critical cryptographic code. A subtle bug here can break
consensus or enable fund theft. Do not introduce timing side-channels.

## Overview

BLS12-381 pairing-friendly elliptic curve implementation. **Fork of
zkcrypto/bls12_381** with Dusk-specific enhancements (multiscalar
multiplication, serde, rkyv, hash-to-scalar, bitwise ops).

All Dusk additions are scoped inside `dusk` submodules (e.g. `src/dusk/`,
`src/scalar/dusk.rs`, `src/g1/dusk.rs`). Do not modify the upstream
zkcrypto code directly.

## Commands

```bash
make test      # Run tests (std + no_std)
make clippy    # Run clippy
make fmt       # Format code (requires nightly)
make check     # Type-check
make doc       # Generate docs
make no-std    # Verify no_std + WASM compatibility
make clean     # Clean build artifacts
cargo bench --bench groups --features groups        # Group benchmarks
cargo bench --bench hash_to_curve --features experimental  # Hash-to-curve benchmarks
```

## Architecture

### Key Files

| Path | Purpose |
|------|---------|
| `src/scalar.rs` | BlsScalar field element (largest file) |
| `src/g1.rs` | G1 curve operations (projective/affine) |
| `src/g2.rs` | G2 curve operations (projective/affine) |
| `src/pairings.rs` | Bilinear pairing implementation |
| `src/fp.rs` | Base field Fp |
| `src/fp2.rs` | Quadratic extension Fp2 |
| `src/fp6.rs` | Cubic extension Fp6 |
| `src/fp12.rs` | Degree-12 extension Fp12 |
| `src/dusk/` | Dusk-specific: multiscalar_mul, choice helpers |
| `src/scalar/dusk.rs` | Dusk-specific scalar ops (serde, Ord, bitwise, hash-to-scalar) |
| `src/g1/dusk.rs` | Dusk-specific G1 ops (serde, rkyv) |
| `src/g2/dusk.rs` | Dusk-specific G2 ops (serde, rkyv) |
| `src/hash_to_curve/` | Hash-to-curve (RFC draft v12) |

### Key Types

- `BlsScalar` (alias `Scalar`) — scalar field element
- `G1Affine` / `G1Projective` — G1 curve points
- `G2Affine` / `G2Projective` — G2 curve points
- `Fp` — base field element
- `Fp2`, `Fp6`, `Fp12` — extension field towers
- Dusk additions: multiscalar multiplication, w-NAF, hash-to-scalar

### Features (defaults: groups, pairings, alloc, bits, parallel, byteorder)

- `groups` — G1, G2, GT arithmetic
- `pairings` — bilinear pairings
- `alloc` — allocator-dependent APIs
- `bits` — bit operations on field elements
- `parallel` — rayon multiscalar multiplication
- `experimental` — hash-to-curve
- `rkyv-impl` — rkyv zero-copy serialization
- `serde` — serde support

## Conventions

- **no_std by default**: the crate is `no_std`. Do not add `std` dependencies.
- **Dusk submodule scoping**: all Dusk additions go in `dusk` submodules
  (`src/dusk/`, `src/scalar/dusk.rs`, `src/g1/dusk.rs`, `src/g2/dusk.rs`).
  Never modify upstream zkcrypto code.
- **No timing side-channels**: do not introduce branches or early returns on
  secret data. Use constant-time operations.
- **Montgomery form**: `Scalar` values are stored in Montgomery form. The
  `Ord`/`PartialOrd` impls compare Montgomery-form limbs, not canonical
  field values.
- **Test both configurations**: always run `make test`, which covers
  `--all-features` and `--no-default-features`.

## Git

Single-crate repo. Commit messages use imperative mood, no scope prefix.

## Changelog

- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change
- Use the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.
- Follow standard markdown formatting: separate headings from surrounding
  content with blank lines, leave a blank line before and after lists, and
  never have two headings back-to-back without a blank line between them
