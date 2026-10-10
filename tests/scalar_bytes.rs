// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::ops::{BitAnd, BitXor};

use dusk_bls12_381::BlsScalar;
use dusk_bytes::{DeserializableSlice, Error, Serializable};

#[test]
fn byte_decoding_requires_a_canonical_scalar() {
    let scalar = -BlsScalar::from(3u64);
    let bytes = <BlsScalar as Serializable<32>>::to_bytes(&scalar);
    assert_eq!(bytes, scalar.to_bytes());
    assert_eq!(
        <BlsScalar as Serializable<32>>::from_bytes(&bytes),
        Ok(scalar)
    );
    assert_eq!(BlsScalar::from_slice(&bytes), Ok(scalar));

    // `-1 + 1` in its little-endian bytes is the modulus itself.
    let mut modulus = (-BlsScalar::one()).to_bytes();
    modulus[0] += 1;
    for bytes in [modulus, [0xff; 32]] {
        assert_eq!(
            <BlsScalar as Serializable<32>>::from_bytes(&bytes),
            Err(Error::InvalidData)
        );
        assert_eq!(BlsScalar::from_slice(&bytes), Err(Error::InvalidData));
    }
}

#[test]
fn bit_operations_act_on_every_canonical_limb() {
    // Both operands, and their bitwise combinations, are below the modulus,
    // and their bits overlap in every limb, so `^`, `|` and `&` all differ.
    let a = BlsScalar::from_raw([
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x0f0f_0f0f_0f0f_0f0f,
        0x00ff_00ff_00ff_00ff,
    ]);
    let b = BlsScalar::from_raw([
        0xffff_0000_ffff_0000,
        0x0000_ffff_0000_ffff,
        0xff00_ff00_ff00_ff00,
        0x0f0f_0f0f_0f0f_0f0f,
    ]);
    let (a_bytes, b_bytes) = (a.to_bytes(), b.to_bytes());
    let xor: [u8; 32] = core::array::from_fn(|i| a_bytes[i] ^ b_bytes[i]);
    let and: [u8; 32] = core::array::from_fn(|i| a_bytes[i] & b_bytes[i]);

    assert_eq!((a ^ b).to_bytes(), xor);
    assert_eq!(BitXor::bitxor(&a, &b).to_bytes(), xor);
    assert_eq!((a & b).to_bytes(), and);
    assert_eq!(BitAnd::bitand(&a, &b).to_bytes(), and);
}
