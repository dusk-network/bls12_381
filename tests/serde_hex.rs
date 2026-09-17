// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(feature = "serde")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Debug;

use dusk_bls12_381::BlsScalar;
use serde::de::value::{
    BorrowedBytesDeserializer, BorrowedStrDeserializer, Error, StringDeserializer,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

thread_local! {
    static ALLOCATED_BYTES: Cell<Option<usize>> = const { Cell::new(None) };
}

struct Allocator;

fn record(size: usize) {
    let _ = ALLOCATED_BYTES.try_with(|allocated| {
        if let Some(previous) = allocated.get() {
            allocated.set(Some(previous.saturating_add(size)));
        }
    });
}

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATED_BYTES.set(Some(0));
    let result = operation();
    // Stop recording before assertions, error formatting, or result disposal.
    let allocated = ALLOCATED_BYTES.replace(None).unwrap();
    (result, allocated)
}

fn check<T: DeserializeOwned + Serialize + Debug + Eq>(value: T) {
    let json = serde_json::to_string(&value).unwrap();
    let hex = &json[1..json.len() - 1];
    let (decoded, allocated) =
        measured(|| T::deserialize(BorrowedStrDeserializer::<Error>::new(hex)));
    assert_eq!(decoded.unwrap(), value);
    assert_eq!(allocated, 0, "valid borrowed hex must not allocate");

    let oversized = "ab".repeat(512 * 1024);
    let (decoded, allocated) =
        measured(|| T::deserialize(BorrowedStrDeserializer::<Error>::new(&oversized)));
    assert!(decoded.is_err());
    assert!(
        allocated < 1024,
        "oversized hex was copied or decoded: {allocated}"
    );

    // Preserve borrowed JSON, reader/owned/escaped strings and byte-backed visitors.
    assert_eq!(serde_json::from_str::<T>(&json).unwrap(), value);
    assert_eq!(
        serde_json::from_reader::<_, T>(json.as_bytes()).unwrap(),
        value
    );
    assert_eq!(
        T::deserialize(StringDeserializer::<Error>::new(hex.to_owned())).unwrap(),
        value
    );
    assert_eq!(
        T::deserialize(BorrowedBytesDeserializer::<Error>::new(hex.as_bytes())).unwrap(),
        value
    );
    let escaped = format!("\"\\u{:04x}{}\"", hex.as_bytes()[0], &hex[1..]);
    assert_eq!(serde_json::from_str::<T>(&escaped).unwrap(), value);
    assert_eq!(
        serde_json::from_str::<T>(&json.to_uppercase()).unwrap(),
        value
    );

    for input in [
        "\"\"".to_owned(),
        format!("\"{}\"", &hex[..hex.len() - 1]),
        format!("\"{hex}00\""),
        format!("\"g{}\"", &hex[1..]),
        format!("\"é{}\"", &hex[2..]),
        format!("\"{}\"", "ff".repeat(hex.len() / 2)),
        "null".to_owned(),
        "[]".to_owned(),
    ] {
        assert!(serde_json::from_str::<T>(&input).is_err(), "{input}");
    }
}

#[test]
fn scalar_hex_decoding_is_bounded() {
    check(BlsScalar::from(7));
    check(BlsScalar::zero());
}

#[cfg(feature = "groups")]
#[test]
fn point_hex_decoding_is_bounded() {
    use dusk_bls12_381::{G1Affine, G2Affine};

    check(G1Affine::generator());
    check(G1Affine::identity());
    check(G2Affine::generator());
    check(G2Affine::identity());
}
