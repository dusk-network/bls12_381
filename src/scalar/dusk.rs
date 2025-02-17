// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use core::cmp::{Ord, Ordering, PartialOrd};
use core::convert::TryFrom;
use core::hash::{Hash, Hasher};
use core::ops::{BitAnd, BitXor};
use dusk_bytes::{Error as BytesError, Serializable};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::{Scalar, MODULUS, R2};
use crate::util::sbb;

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Scalar) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scalar {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..4).rev() {
            #[allow(clippy::comparison_chain)]
            if self.0[i] > other.0[i] {
                return Ordering::Greater;
            } else if self.0[i] < other.0[i] {
                return Ordering::Less;
            }
        }
        Ordering::Equal
    }
}

impl Serializable<32> for Scalar {
    type Error = BytesError;

    /// Converts an element of `Scalar` into a byte representation in
    /// little-endian byte order.
    fn to_bytes(&self) -> [u8; Self::SIZE] {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = Scalar::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);

        let mut res = [0; Self::SIZE];
        res[0..8].copy_from_slice(&tmp.0[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp.0[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp.0[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp.0[3].to_le_bytes());

        res
    }

    /// Attempts to convert a little-endian byte representation of
    /// a scalar into a `Scalar`, failing if the input is not canonical.
    fn from_bytes(buf: &[u8; Self::SIZE]) -> Result<Self, Self::Error> {
        let mut s = [0u64; 4];

        s.iter_mut()
            .zip(buf.chunks_exact(8))
            .try_for_each(|(s, b)| {
                <[u8; 8]>::try_from(b)
                    .map(|b| *s = u64::from_le_bytes(b))
                    .map_err(|_| BytesError::InvalidData)
            })?;

        // Try to subtract the modulus
        let (_, borrow) = sbb(s[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(s[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(s[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(s[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        if (borrow as u8) & 1 != 1 {
            return Err(BytesError::InvalidData);
        }

        let mut s = Scalar(s);

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        s *= &R2;

        Ok(s)
    }
}

#[cfg(feature = "serde")]
mod serde_support {
    extern crate alloc;

    use alloc::string::{String, ToString};

    use serde::de::Error as SerdeError;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl Serialize for Scalar {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let s = hex::encode(self.to_bytes());
            s.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Scalar {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let s = String::deserialize(deserializer)?;
            let decoded = hex::decode(&s).map_err(SerdeError::custom)?;
            let decoded_len = decoded.len();
            let bytes: [u8; Scalar::SIZE] = decoded.try_into().map_err(|_| {
                SerdeError::invalid_length(decoded_len, &Scalar::SIZE.to_string().as_str())
            })?;
            let scalar = Scalar::from_bytes(&bytes)
                .into_option()
                .ok_or(SerdeError::custom(
                    "Failed to deserialize Scalar: invalid Scalar",
                ))?;
            Ok(scalar)
        }
    }

    #[test]
    fn serde_scalar() {
        use ff::Field;
        use rand::rngs::StdRng;
        use rand_core::SeedableRng;

        let mut rng = StdRng::seed_from_u64(0xc0b);
        let scalar = Scalar::random(&mut rng);
        let ser = serde_json::to_string(&scalar).unwrap();
        let deser = serde_json::from_str(&ser).unwrap();

        assert_eq!(scalar, deser);
    }

    #[test]
    fn serde_scalar_too_short_encoded() {
        let length_31_enc = "\"fe9a9c1876745ca351435dec31217662ff1fcf67287de6fd9b6c7de1d0846b\"";

        let scalar: Result<Scalar, _> = serde_json::from_str(&length_31_enc);
        assert!(scalar.is_err());
    }

    #[test]
    fn serde_scalar_too_long_encoded() {
        let length_33_enc =
            "\"fe9a9c1876745ca351435dec31217662ff1fcf67287de6fd9b6c7de1d0846b2100\"";

        let scalar: Result<Scalar, _> = serde_json::from_str(&length_33_enc);
        assert!(scalar.is_err());
    }
}

#[allow(dead_code)]
pub const GEN_X: Scalar = Scalar([
    0x1539098E9CBCC1D5,
    0x0CCC77B0E1804E8D,
    0x6EEF947A6FD0FB2C,
    0xA3D063F54E10DDE9,
]);

#[allow(dead_code)]
pub const GEN_Y: Scalar = Scalar([
    0x6540D21E7007DC60,
    0x3B0D848E832A862F,
    0xB53BB87E05DA8257,
    0xCD482CC3FD6FF4D,
]);

impl<'a, 'b> BitXor<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn bitxor(self, rhs: &'b Scalar) -> Scalar {
        let a_red = self.reduce();
        let b_red = rhs.reduce();
        Scalar::from_raw([
            a_red.0[0] ^ b_red.0[0],
            a_red.0[1] ^ b_red.0[1],
            a_red.0[2] ^ b_red.0[2],
            a_red.0[3] ^ b_red.0[3],
        ])
    }
}

impl BitXor<Scalar> for Scalar {
    type Output = Scalar;

    fn bitxor(self, rhs: Scalar) -> Scalar {
        &self ^ &rhs
    }
}

impl BitAnd<Scalar> for Scalar {
    type Output = Scalar;

    fn bitand(self, rhs: Scalar) -> Scalar {
        &self & &rhs
    }
}

impl<'a, 'b> BitAnd<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn bitand(self, rhs: &'b Scalar) -> Scalar {
        let a_red = self.reduce();
        let b_red = rhs.reduce();
        Scalar::from_raw([
            a_red.0[0] & b_red.0[0],
            a_red.0[1] & b_red.0[1],
            a_red.0[2] & b_red.0[2],
            a_red.0[3] & b_red.0[3],
        ])
    }
}

impl Hash for Scalar {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Scalar {
    /// Checks in ct_time whether a Scalar is equal to zero.
    pub fn is_zero(&self) -> Choice {
        self.ct_eq(&Scalar::zero())
    }

    /// Checks in ct_time whether a Scalar is equal to one.   
    pub fn is_one(&self) -> Choice {
        self.ct_eq(&Scalar::one())
    }

    /// Returns the internal representation of the Scalar.
    pub const fn internal_repr(&self) -> &[u64; 4] {
        &self.0
    }

    /// Returns the bit representation of the given `Scalar` as
    /// an array of 256 bits represented as `u8`.
    pub fn to_bits(&self) -> [u8; 256] {
        let mut res = [0u8; 256];
        let bytes = self.to_bytes();
        for (byte, bits) in bytes.iter().zip(res.chunks_mut(8)) {
            bits.iter_mut()
                .enumerate()
                .for_each(|(i, bit)| *bit = (byte >> i) & 1)
        }
        res
    }

    /// Converts an element of `Scalar` into a byte representation in
    /// big-endian byte order.
    pub fn to_be_bytes(&self) -> [u8; Self::SIZE] {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = self.reduce();

        let mut res = [0; Self::SIZE];
        res[0..8].copy_from_slice(&tmp.0[3].to_be_bytes());
        res[8..16].copy_from_slice(&tmp.0[2].to_be_bytes());
        res[16..24].copy_from_slice(&tmp.0[1].to_be_bytes());
        res[24..32].copy_from_slice(&tmp.0[0].to_be_bytes());

        res
    }

    /// Reduces the scalar and returns it multiplied by the montgomery
    /// radix.
    pub fn reduce(&self) -> Scalar {
        Scalar::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0)
    }

    /// Computes `2^X` where X is a `u64` without the need to generate
    /// an array in the stack as `pow` & `pow_vartime` require.
    pub fn pow_of_2(by: u64) -> Self {
        let two = Scalar::from(2u64);
        let mut res = Self::one();
        for i in (0..64).rev() {
            res = res.square();
            let mut tmp = res;
            tmp *= two;
            res.conditional_assign(&tmp, (((by >> i) & 0x1) as u8).into());
        }
        res
    }

    /// Creates a `Scalar` from arbitrary bytes by hashing the input with
    /// BLAKE2b into a 512-bits number, and then converting the number into its
    /// `Scalar` representation by reducing it by the modulo.
    ///
    /// By treating the output of the BLAKE2b hash as a random oracle, this
    /// implementation follows the first conversion of
    /// https://hackmd.io/zV6qe1_oSU-kYU6Tt7pO7Q with concrete numbers:
    /// ```text
    /// p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
    /// p = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    ///
    /// s = 2^512
    /// s = 13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084096
    ///
    /// r = 255699135089535202043525422716183576215815630510683217819334674386498370757523
    ///
    /// m = 3294906474794265442129797520630710739278575682199800681788903916070560242797
    /// ```
    pub fn hash_to_scalar(input: &[u8]) -> Scalar {
        let state = blake2b_simd::Params::new()
            .hash_length(64)
            .to_state()
            .update(input)
            .finalize();

        let bytes = state.as_bytes();

        Scalar::from_u512([
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[32..40]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[40..48]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[48..56]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[56..64]).unwrap()),
        ])
    }

    /// SHR impl
    #[inline]
    pub fn divn(&mut self, mut n: u32) {
        if n >= 256 {
            *self = Self::from(0);
            return;
        }

        while n >= 64 {
            let mut t = 0;
            for i in self.0.iter_mut().rev() {
                core::mem::swap(&mut t, i);
            }
            n -= 64;
        }

        if n > 0 {
            let mut t = 0;
            for i in self.0.iter_mut().rev() {
                let t2 = *i << (64 - n);
                *i >>= n;
                *i |= t;
                t = t2;
            }
        }
    }
}

#[test]
fn test_partial_ord() {
    let one = Scalar::one();
    assert!(one < -one);
}

#[test]
fn test_xor() {
    let a = Scalar::from(500u64);
    let b = Scalar::from(499u64);
    let res = Scalar::from(7u64);
    assert_eq!(&a ^ &b, res);
}

#[test]
fn test_and() {
    let a = Scalar::one();
    let b = Scalar::one();
    let res = Scalar::one();
    assert_eq!(&a & &b, res);
    assert_eq!(a & -a, Scalar::zero());
}

#[test]
fn test_iter_sum() {
    let scalars = vec![Scalar::one(), Scalar::one()];
    let res: Scalar = scalars.iter().sum();
    assert_eq!(res, Scalar::one() + Scalar::one());
}

#[test]
fn test_iter_prod() {
    let scalars = vec![Scalar::one() + Scalar::one(), Scalar::one() + Scalar::one()];
    let res: Scalar = scalars.iter().product();
    assert_eq!(res, Scalar::from(4u64));
}

#[test]
fn bit_repr() {
    let two_pow_128 = Scalar::from(2u64).pow(&[128, 0, 0, 0]);
    let two_pow_128_bits = [
        0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(&two_pow_128.to_bits()[..], &two_pow_128_bits[..]);

    let two_pow_128_minus_rand = Scalar::from(2u64).pow(&[128, 0, 0, 0]) - Scalar::from(7568589u64);
    let two_pow_128_bits = [
        1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
    ];
    assert_eq!(
        &two_pow_128_minus_rand.to_bits()[..128],
        &two_pow_128_bits[..]
    )
}

#[test]
fn pow_of_two_test() {
    let two = Scalar::from(2u64);
    for i in 0..1000 {
        assert_eq!(Scalar::pow_of_2(i as u64), two.pow(&[i as u64, 0, 0, 0]));
    }
}

#[test]
fn test_scalar_eq_and_hash() {
    use sha3::{Digest, Keccak256};

    let r0 = Scalar::from_raw([
        0x1fff_3231_233f_fffd,
        0x4884_b7fa_0003_4802,
        0x998c_4fef_ecbc_4ff3,
        0x1824_b159_acc5_0562,
    ]);
    let r1 = Scalar::from_raw([
        0x1fff_3231_233f_fffd,
        0x4884_b7fa_0003_4802,
        0x998c_4fef_ecbc_4ff3,
        0x1824_b159_acc5_0562,
    ]);
    let r2 = Scalar::from(7);

    // Check PartialEq
    assert!(r0 == r1);
    assert!(r0 != r2);

    let hash_r0 = Keccak256::digest(&r0.to_bytes());
    let hash_r1 = Keccak256::digest(&r1.to_bytes());
    let hash_r2 = Keccak256::digest(&r2.to_bytes());

    // Check if hash results are consistent with PartialEq results
    assert_eq!(hash_r0, hash_r1);
    assert_ne!(hash_r0, hash_r2);
}

#[test]
fn test_to_be_bytes() {
    assert_eq!(
        Scalar::zero().to_be_bytes(),
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ]
    );

    assert_eq!(
        Scalar::one().to_be_bytes(),
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1
        ]
    );

    assert_eq!(
        R2.to_be_bytes(),
        [
            24, 36, 177, 89, 172, 197, 5, 111, 153, 140, 79, 239, 236, 188, 79, 245, 88, 132, 183,
            250, 0, 3, 72, 2, 0, 0, 0, 1, 255, 255, 255, 254
        ]
    );

    assert_eq!(
        (-&Scalar::one()).to_be_bytes(),
        [
            115, 237, 167, 83, 41, 157, 125, 72, 51, 57, 216, 8, 9, 161, 216, 5, 83, 189, 164, 2,
            255, 254, 91, 254, 255, 255, 255, 255, 0, 0, 0, 0
        ]
    );
}

#[cfg(all(test, feature = "alloc"))]
mod fuzz {
    use alloc::vec::Vec;

    use crate::scalar::{Scalar, MODULUS};
    use crate::util::sbb;

    fn is_scalar_in_range(scalar: &Scalar) -> bool {
        // subtraction against modulus must underflow
        let borrow = scalar
            .0
            .iter()
            .zip(MODULUS.0.iter())
            .fold(0, |borrow, (&s, &m)| sbb(s, m, borrow).1);

        borrow == u64::MAX
    }

    quickcheck::quickcheck! {
        fn prop_scalar_from_raw_bytes(bytes: Vec<u8>) -> bool {
            let scalar = Scalar::hash_to_scalar(&bytes);

            is_scalar_in_range(&scalar)
        }
    }
}
