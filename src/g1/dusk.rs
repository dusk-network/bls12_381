// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use dusk_bytes::{Error as BytesError, Serializable};
use subtle::{ConstantTimeEq, CtOption};

use super::G1Affine;
use crate::fp::Fp;

impl G1Affine {
    /// Bytes size of the raw representation
    pub const RAW_SIZE: usize = 97;

    /// Raw bytes representation
    ///
    /// The intended usage of this function is for trusted sets of data where performance is
    /// critical.
    ///
    /// For secure serialization, check `to_bytes`
    pub fn to_raw_bytes(&self) -> [u8; Self::RAW_SIZE] {
        let mut bytes = [0u8; Self::RAW_SIZE];
        let chunks = bytes.chunks_mut(8);

        self.x
            .internal_repr()
            .iter()
            .chain(self.y.internal_repr().iter())
            .zip(chunks)
            .for_each(|(n, c)| c.copy_from_slice(&n.to_le_bytes()));

        bytes[Self::RAW_SIZE - 1] = self.infinity.into();

        bytes
    }

    /// Attempts to create a `G1Affine` from its raw representation.
    ///
    /// The coordinates must be canonical field elements and the resulting point
    /// must be on the curve and in the prime-order subgroup.
    pub fn from_slice(bytes: &[u8]) -> CtOption<Self> {
        let valid_length = subtle::Choice::from((bytes.len() == Self::RAW_SIZE) as u8);
        let mut raw = [0u8; Self::RAW_SIZE];
        if bool::from(valid_length) {
            raw.copy_from_slice(bytes);
        }

        // SAFETY: `raw` has the exact size expected by the unchecked decoder.
        let point = unsafe { Self::from_slice_unchecked(&raw) };
        let infinity = raw[Self::RAW_SIZE - 1];
        let valid_infinity = infinity.ct_eq(&0) | infinity.ct_eq(&1);
        let canonical_identity =
            infinity.ct_eq(&0) | (point.x.is_zero() & point.y.ct_eq(&Fp::one()));

        CtOption::new(
            point,
            valid_length
                & point.x.is_canonical()
                & point.y.is_canonical()
                & valid_infinity
                & canonical_identity
                & point.is_on_curve()
                & point.is_torsion_free(),
        )
    }

    /// Create a `G1Affine` from a set of bytes created by `G1Affine::to_raw_bytes`.
    ///
    /// # Safety
    /// No check is performed and no constant time is granted. The expected
    /// usage of this function is for trusted bytes where performance is critical.
    /// For checked raw decoding, use [`G1Affine::from_slice`]. For canonical
    /// compressed decoding, use `from_bytes`.
    pub unsafe fn from_slice_unchecked(bytes: &[u8]) -> Self {
        let mut x = [0u64; 6];
        let mut y = [0u64; 6];
        let mut z = [0u8; 8];

        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .zip(x.iter_mut().chain(y.iter_mut()))
            .for_each(|(c, n)| {
                z.copy_from_slice(c);
                *n = u64::from_le_bytes(z);
            });

        let x = Fp::from_raw_unchecked(x);
        let y = Fp::from_raw_unchecked(y);

        let infinity = if bytes.len() >= Self::RAW_SIZE {
            bytes[Self::RAW_SIZE - 1].into()
        } else {
            0u8.into()
        };

        Self { x, y, infinity }
    }
}

impl Serializable<48> for G1Affine {
    type Error = BytesError;

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        self.to_compressed()
    }

    fn from_bytes(buf: &[u8; Self::SIZE]) -> Result<Self, Self::Error> {
        Option::from(Self::from_compressed(buf)).ok_or(BytesError::InvalidData)
    }
}

#[cfg(feature = "serde")]
mod serde_support {
    extern crate alloc;

    use alloc::format;
    use alloc::string::{String, ToString};

    use serde::de::Error as SerdeError;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl Serialize for G1Affine {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let s = hex::encode(self.to_bytes());
            s.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for G1Affine {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let s = String::deserialize(deserializer)?;
            let decoded = hex::decode(&s).map_err(SerdeError::custom)?;
            let decoded_len = decoded.len();
            let bytes: [u8; G1Affine::SIZE] = decoded.try_into().map_err(|_| {
                SerdeError::invalid_length(decoded_len, &G1Affine::SIZE.to_string().as_str())
            })?;
            let affine = G1Affine::from_bytes(&bytes)
                .map_err(|err| SerdeError::custom(format!("{err:?}")))?;
            Ok(affine)
        }
    }

    #[cfg(test)]
    mod tests {
        use alloc::boxed::Box;

        use super::*;
        use crate::dusk::test_utils;

        #[test]
        fn serde_g1_affine() -> Result<(), Box<dyn std::error::Error>> {
            let gen = G1Affine::generator();
            let ser = test_utils::assert_canonical_json(
                &gen,
                "\"97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb\""
            )?;
            let deser: G1Affine = serde_json::from_str(&ser).unwrap();
            assert_eq!(gen, deser);
            Ok(())
        }

        #[test]
        fn serde_g1_affine_too_short_encoded() {
            let length_47_enc = "\"97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6\"";

            let g1_affine: Result<G1Affine, _> = serde_json::from_str(&length_47_enc);
            assert!(g1_affine.is_err());
        }

        #[test]
        fn serde_g1_affine_too_long_encoded() {
            let length_49_enc = "\"97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb00\"";

            let g1_affine: Result<G1Affine, _> = serde_json::from_str(&length_49_enc);
            assert!(g1_affine.is_err());
        }
    }
}

#[test]
fn g1_affine_bytes_unchecked() {
    let gen = G1Affine::generator();
    let ident = G1Affine::identity();

    let gen_p = gen.to_raw_bytes();
    let gen_p = unsafe { G1Affine::from_slice_unchecked(&gen_p) };

    let ident_p = ident.to_raw_bytes();
    let ident_p = unsafe { G1Affine::from_slice_unchecked(&ident_p) };

    assert_eq!(gen, gen_p);
    assert_eq!(ident, ident_p);
}

#[test]
fn g1_affine_bytes_checked() {
    for point in [G1Affine::generator(), G1Affine::identity()] {
        let decoded = G1Affine::from_slice(&point.to_raw_bytes());
        assert_eq!(Option::<G1Affine>::from(decoded), Some(point));
    }

    let raw = G1Affine::generator().to_raw_bytes();
    assert!(bool::from(G1Affine::from_slice(&raw[..96]).is_none()));

    let mut too_long = [0u8; G1Affine::RAW_SIZE + 1];
    too_long[..G1Affine::RAW_SIZE].copy_from_slice(&raw);
    assert!(bool::from(G1Affine::from_slice(&too_long).is_none()));

    let mut invalid_infinity = raw;
    invalid_infinity[G1Affine::RAW_SIZE - 1] = 2;
    assert!(bool::from(
        G1Affine::from_slice(&invalid_infinity).is_none()
    ));

    let mut noncanonical_identity = G1Affine::identity().to_raw_bytes();
    noncanonical_identity[0] = 1;
    assert!(bool::from(
        G1Affine::from_slice(&noncanonical_identity).is_none()
    ));

    let off_curve = [0u8; G1Affine::RAW_SIZE];
    assert!(bool::from(G1Affine::from_slice(&off_curve).is_none()));
}

#[test]
fn g1_affine_bytes_checked_reject_noncanonical_field() {
    const MODULUS: [u64; 6] = [
        0xb9fe_ffff_ffff_aaab,
        0x1eab_fffe_b153_ffff,
        0x6730_d2a0_f6b0_f624,
        0x6477_4b84_f385_12bf,
        0x4b1b_a7b6_434b_acd7,
        0x1a01_11ea_397f_e69a,
    ];

    let mut point = G1Affine::generator();
    let mut x = *point.x.internal_repr();
    let mut carry = 0;
    for (limb, modulus) in x.iter_mut().zip(MODULUS) {
        (*limb, carry) = crate::util::adc(*limb, modulus, carry);
    }
    assert_eq!(carry, 0);

    point.x = Fp::from_raw_unchecked(x);
    assert!(bool::from(point.is_on_curve()));
    assert!(bool::from(point.is_torsion_free()));
    assert!(bool::from(
        G1Affine::from_slice(&point.to_raw_bytes()).is_none()
    ));
}

#[test]
fn g1_affine_bytes_checked_reject_wrong_subgroup() {
    let point = G1Affine {
        x: Fp::from_raw_unchecked([
            0x0aba_f895_b97e_43c8,
            0xba4c_6432_eb9b_61b0,
            0x1250_6f52_adfe_307f,
            0x7502_8c34_3933_6b72,
            0x8474_4f05_b8e9_bd71,
            0x113d_554f_b095_54f7,
        ]),
        y: Fp::from_raw_unchecked([
            0x73e9_0e88_f5cf_01c0,
            0x3700_7b65_dd31_97e2,
            0x5cf9_a199_2f0d_7c78,
            0x4f83_c10b_9eb3_330d,
            0xf6a6_3f6f_07f6_0961,
            0x0c53_b5b9_7e63_4df3,
        ]),
        infinity: 0u8.into(),
    };

    assert!(bool::from(point.is_on_curve()));
    assert!(!bool::from(point.is_torsion_free()));
    assert!(bool::from(
        G1Affine::from_slice(&point.to_raw_bytes()).is_none()
    ));
}

#[test]
fn g1_affine_bytes_unchecked_field() {
    let x = Fp::from_raw_unchecked([
        0x9af1f35780fffb82,
        0x557416ceeea5a52f,
        0x1e4403e4911a2d97,
        0xb85bfb438316bf2,
        0xa3b716c69a9e5a7b,
        0x1fe9b8ad976dd39,
    ]);

    let y = Fp::from_raw_unchecked([
        0xb4f1cc806acfb4e2,
        0x38c28cba4cf600ed,
        0x3af1c2f54a01a366,
        0x96a75ac708a9eb72,
        0x4253bd59228e50d,
        0x120114fae4294c21,
    ]);

    let infinity = 0u8.into();
    let g = G1Affine { x, y, infinity };

    let g_p = g.to_raw_bytes();
    let g_p = unsafe { G1Affine::from_slice_unchecked(&g_p) };

    assert_eq!(g, g_p);
}
