// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(all(
    feature = "groups",
    feature = "rkyv-impl",
    not(feature = "rkyv-semantic-validation")
))]

use bytecheck::CheckBytes;
#[cfg(all(feature = "pairings", feature = "alloc"))]
use dusk_bls12_381::G2Prepared;
use dusk_bls12_381::{G1Affine, G1Projective, G2Affine, G2Projective};
#[cfg(feature = "pairings")]
use dusk_bls12_381::{Gt, MillerLoopResult};
use rkyv::validation::validators::DefaultValidator;
use rkyv::Archive;

fn assert_legacy_check_bytes<T>()
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<DefaultValidator<'a>>,
{
}

#[test]
fn structural_mode_retains_existing_check_bytes_bounds() {
    assert_legacy_check_bytes::<G1Affine>();
    assert_legacy_check_bytes::<G1Projective>();
    assert_legacy_check_bytes::<G2Affine>();
    assert_legacy_check_bytes::<G2Projective>();
    #[cfg(all(feature = "pairings", feature = "alloc"))]
    assert_legacy_check_bytes::<G2Prepared>();
    #[cfg(feature = "pairings")]
    assert_legacy_check_bytes::<MillerLoopResult>();
    #[cfg(feature = "pairings")]
    assert_legacy_check_bytes::<Gt>();
}

#[test]
fn structural_mode_preserves_legacy_nested_point_decoding() {
    // SAFETY: the exact-size raw representations are deliberately used to
    // construct canonical field elements that do not represent curve points.
    let g1 = unsafe { G1Affine::from_slice_unchecked(&[0; G1Affine::RAW_SIZE]) };
    let g2 = unsafe { G2Affine::from_slice_unchecked(&[0; G2Affine::RAW_SIZE]) };
    assert!(!bool::from(g1.is_on_curve()));
    assert!(!bool::from(g2.is_on_curve()));

    let value = (vec![7u8; 32], vec![g2; 10], g1);
    let bytes = rkyv::to_bytes::<_, 1024>(&value).unwrap();
    let (_, decoded_keys, decoded_signature) =
        rkyv::from_bytes::<(Vec<u8>, Vec<G2Affine>, G1Affine)>(&bytes).unwrap();

    assert_eq!(decoded_keys.len(), 10);
    assert!(!bool::from(decoded_keys[0].is_on_curve()));
    assert!(!bool::from(decoded_signature.is_on_curve()));
}
