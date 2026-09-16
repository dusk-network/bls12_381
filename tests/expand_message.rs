#![cfg(feature = "experimental")]

use dusk_bls12_381::hash_to_curve::{
    ExpandMessage, ExpandMessageState, ExpandMsgXmd, ExpandMsgXof, HashToField, InitExpandMessage,
};
use sha2::{Sha256, Sha512};
#[allow(deprecated)] // The experimental HashToField trait uses digest 0.9's GenericArray.
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

#[test]
fn xof_rejects_unsupported_lengths() {
    for len in [65536, 65568, usize::MAX] {
        assert!(std::panic::catch_unwind(|| {
            ExpandMsgXof::<Shake128>::init_expand(b"message", b"DST", len)
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            ExpandMsgXof::<Shake256>::init_expand(b"message", b"DST", len)
        })
        .is_err());
    }
}

#[test]
fn xmd_rejects_unsupported_lengths_without_wrapping() {
    for len in [255 * 32 + 1, 65536, usize::MAX] {
        assert!(std::panic::catch_unwind(|| {
            ExpandMsgXmd::<Sha256>::init_expand(b"message", b"DST", len)
        })
        .is_err());
    }
    assert!(std::panic::catch_unwind(|| {
        ExpandMsgXmd::<Sha512>::init_expand(b"message", b"DST", 255 * 64 + 1)
    })
    .is_err());
}

#[test]
#[should_panic(expected = "hash_to_field output length overflows usize")]
#[allow(deprecated)]
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
