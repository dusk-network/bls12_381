// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(any(feature = "hash-to-curve", feature = "experimental"))]

use dusk_bls12_381::hash_to_curve::{
    ExpandMessage, ExpandMessageState, ExpandMsgXmd, ExpandMsgXof, HashToField,
};
use sha2::{Sha256, Sha512};
#[allow(deprecated)] // The HashToField API retains digest 0.9's GenericArray.
use sha3::digest::generic_array::{typenum::U64, GenericArray};
use sha3::{Shake128, Shake256};

fn check_stream<X: ExpandMessage>(len: usize) {
    let mut expected = vec![0; len];
    assert_eq!(
        X::init_expand(b"message", b"DST", len).read_into(&mut expected),
        len
    );
    let mut streamed = X::init_expand(b"message", b"DST", len);
    assert_eq!(streamed.read_into(&mut []), 0);
    assert_eq!(streamed.remain(), len);
    let mut actual = vec![0; len];
    for chunk in actual.chunks_mut(7) {
        assert_eq!(streamed.read_into(chunk), chunk.len());
    }
    assert_eq!(streamed.remain(), 0);
    assert_eq!(streamed.read_into(&mut [0; 7]), 0);
    assert_eq!(actual, expected);
}

#[test]
fn supported_lengths_preserve_streaming() {
    for len in [0, 1, 32, 65535] {
        check_stream::<ExpandMsgXof<Shake128>>(len);
        check_stream::<ExpandMsgXof<Shake256>>(len);
    }
    for len in [0, 1, 32, 255 * 32] {
        check_stream::<ExpandMsgXmd<Sha256>>(len);
    }
    check_stream::<ExpandMsgXmd<Sha512>>(255 * 64);
}

fn check_length_panic<X: ExpandMessage>(len: usize, expected: &str) {
    let panic = std::panic::catch_unwind(|| {
        X::init_expand(b"message", b"DST", len);
    })
    .expect_err("unsupported expansion length must panic");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .expect("panic payload must be a string");
    assert!(
        message.contains(expected),
        "expected panic containing {expected:?}, got {message:?}"
    );
}

#[test]
fn xof_rejects_unsupported_lengths() {
    for len in [65536, 65568, usize::MAX] {
        let expected = "Invalid ExpandMsgXof usage: len_in_bytes > 65535";
        check_length_panic::<ExpandMsgXof<Shake128>>(len, expected);
        check_length_panic::<ExpandMsgXof<Shake256>>(len, expected);
    }
}

#[test]
fn xmd_rejects_unsupported_lengths_without_wrapping() {
    for len in [65536, usize::MAX] {
        check_length_panic::<ExpandMsgXmd<Sha256>>(
            len,
            "Invalid ExpandMsgXmd usage: len_in_bytes > 65535",
        );
    }
    check_length_panic::<ExpandMsgXmd<Sha256>>(255 * 32 + 1, "ell > 255");
    check_length_panic::<ExpandMsgXmd<Sha512>>(255 * 64 + 1, "ell > 255");
}

#[test]
#[should_panic(expected = "hash_to_field output length overflows usize")]
#[allow(deprecated)] // Exercise the existing GenericArray-based HashToField API.
fn hash_to_field_rejects_length_overflow_before_expansion() {
    // A zero-sized field exercises the length arithmetic without a huge allocation.
    #[derive(Clone, Copy)]
    struct Field;
    impl HashToField for Field {
        type InputLength = U64;

        fn from_okm(_: &GenericArray<u8, U64>) -> Self {
            panic!("overflow reached field conversion");
        }
    }
    let mut output = [Field; usize::MAX / 64 + 1];
    Field::hash_to_field::<ExpandMsgXof<Shake128>>(b"message", b"DST", &mut output);
}
