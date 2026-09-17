// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use dusk_bytes::{Error as BytesError, Serializable};

use super::G2Affine;
use crate::fp::Fp;
use crate::fp2::Fp2;

#[cfg(feature = "rkyv-validation")]
crate::dusk::archive::points::checked_point!(
    G2Affine,
    super::ArchivedG2Affine,
    [x, y, infinity],
    |point| point.is_valid()
);

#[cfg(feature = "rkyv-validation")]
crate::dusk::archive::points::checked_point!(
    crate::G2Projective,
    super::ArchivedG2Projective,
    [x, y, z],
    |point| {
        let canonical_identity = !point.is_identity() | (point.x.is_zero() & !point.y.is_zero());
        canonical_identity & G2Affine::from(point).is_valid()
    }
);

impl G2Affine {
    /// Check curve, subgroup and canonical affine identity semantics.
    #[cfg(feature = "rkyv-validation")]
    pub(crate) fn is_valid(&self) -> subtle::Choice {
        use subtle::ConstantTimeEq;

        let canonical_identity =
            !self.is_identity() | (self.x.is_zero() & self.y.ct_eq(&Fp2::one()));
        canonical_identity & self.is_on_curve() & self.is_torsion_free()
    }

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
            .c0
            .internal_repr()
            .iter()
            .chain(self.x.c1.internal_repr().iter())
            .chain(self.y.c0.internal_repr().iter())
            .chain(self.y.c1.internal_repr().iter())
            .zip(chunks)
            .for_each(|(n, c)| c.copy_from_slice(&n.to_le_bytes()));

        bytes[Self::RAW_SIZE - 1] = self.infinity.into();

        bytes
    }

    /// Create a `G2Affine` from a set of bytes created by `G2Affine::to_raw_bytes`.
    ///
    /// # Safety
    /// No check is performed and no constant time is granted. The expected
    /// usage of this function is for trusted bytes where performance is
    /// critical.
    /// For secure serialization, check `from_bytes`.
    /// After generating the point, you can check `is_on_curve` and
    /// `is_torsion_free` to grant its security.
    pub unsafe fn from_slice_unchecked(bytes: &[u8]) -> Self {
        let mut xc0 = [0u64; 6];
        let mut xc1 = [0u64; 6];
        let mut yc0 = [0u64; 6];
        let mut yc1 = [0u64; 6];
        let mut z = [0u8; 8];

        xc0.iter_mut()
            .chain(xc1.iter_mut())
            .chain(yc0.iter_mut())
            .chain(yc1.iter_mut())
            .zip(bytes.as_chunks::<8>().0.iter())
            .for_each(|(n, c)| {
                z.copy_from_slice(c);
                *n = u64::from_le_bytes(z);
            });

        let c0 = Fp::from_raw_unchecked(xc0);
        let c1 = Fp::from_raw_unchecked(xc1);
        let x = Fp2 { c0, c1 };

        let c0 = Fp::from_raw_unchecked(yc0);
        let c1 = Fp::from_raw_unchecked(yc1);
        let y = Fp2 { c0, c1 };

        let infinity = if bytes.len() >= Self::RAW_SIZE {
            bytes[Self::RAW_SIZE - 1].into()
        } else {
            0u8.into()
        };

        Self { x, y, infinity }
    }
}

impl Serializable<96> for G2Affine {
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

    use serde::de::Error as SerdeError;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;
    use crate::dusk::serde::deserialize_hex;

    impl Serialize for G2Affine {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let s = hex::encode(self.to_bytes());
            s.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for G2Affine {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let bytes = deserialize_hex(deserializer)?;
            let affine = G2Affine::from_bytes(&bytes)
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
        fn serde_g2_affine() -> Result<(), Box<dyn std::error::Error>> {
            let gen = G2Affine::generator();
            let ser = test_utils::assert_canonical_json(
                &gen,
                "\"93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8\""
            )?;
            let deser: G2Affine = serde_json::from_str(&ser).unwrap();
            assert_eq!(gen, deser);
            Ok(())
        }

        #[test]
        fn serde_g2_affine_too_short_encoded() {
            let length_95_enc: &str = "\"93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bd\"";

            let g2_affine: Result<G2Affine, _> = serde_json::from_str(&length_95_enc);
            assert!(g2_affine.is_err());
        }

        #[test]
        fn serde_g2_affine_too_long_encoded() {
            let length_97_enc = "\"93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb800\"";

            let g2_affine: Result<G2Affine, _> = serde_json::from_str(&length_97_enc);
            assert!(g2_affine.is_err());
        }
    }
}

#[test]
fn g2_affine_serializable_rejects_malformed_encodings() {
    let from_bytes = <G2Affine as Serializable<96>>::from_bytes;
    for point in [G2Affine::generator(), G2Affine::identity()] {
        assert_eq!(from_bytes(&point.to_bytes()), Ok(point));
    }

    // Same affine point as g2::test_is_torsion_free.
    let wrong_subgroup = G2Affine {
        x: Fp2 {
            c0: Fp::from_raw_unchecked([
                0x89f5_50c8_13db_6431,
                0xa50b_e8c4_56cd_8a1a,
                0xa45b_3741_14ca_e851,
                0xbb61_90f5_bf7f_ff63,
                0x970c_a02c_3ba8_0bc7,
                0x02b8_5d24_e840_fbac,
            ]),
            c1: Fp::from_raw_unchecked([
                0x6888_bc53_d707_16dc,
                0x3dea_6b41_1768_2d70,
                0xd8f5_f930_500c_a354,
                0x6b5e_cb65_56f5_c155,
                0xc96b_ef04_3477_8ab0,
                0x0508_1505_5150_06ad,
            ]),
        },
        y: Fp2 {
            c0: Fp::from_raw_unchecked([
                0x3cf1_ea0d_434b_0f40,
                0x1a0d_c610_e603_e333,
                0x7f89_9561_60c7_2fa0,
                0x25ee_03de_cf64_31c5,
                0xeee8_e206_ec0f_e137,
                0x0975_92b2_26df_ef28,
            ]),
            c1: Fp::from_raw_unchecked([
                0x71e8_bb5f_2924_7367,
                0xa5fe_049e_2118_31ce,
                0x0ce6_b354_502a_3896,
                0x93b0_1200_0997_314e,
                0x6759_f3b6_aa5b_42ac,
                0x1569_44c4_dfe9_2bbb,
            ]),
        },
        infinity: 0u8.into(),
    };
    assert!(bool::from(wrong_subgroup.is_on_curve()));
    assert!(!bool::from(wrong_subgroup.is_torsion_free()));
    let wrong_subgroup_bytes = wrong_subgroup.to_compressed();
    assert_eq!(
        Option::<G2Affine>::from(G2Affine::from_compressed_unchecked(&wrong_subgroup_bytes)),
        Some(wrong_subgroup)
    );

    let mut no_compression = G2Affine::generator().to_bytes();
    no_compression[0] &= !0x80;
    let mut nonzero_infinity = G2Affine::identity().to_bytes();
    nonzero_infinity[95] = 1;

    for (case, bytes) in [
        ("cleared compression flag", no_compression),
        ("infinity with nonzero x", nonzero_infinity),
        ("all-zero buffer", [0; 96]),
        ("point outside prime-order subgroup", wrong_subgroup_bytes),
    ] {
        assert_eq!(from_bytes(&bytes), Err(BytesError::InvalidData), "{case}");
    }
}

#[test]
fn g2_affine_bytes_unchecked() {
    let gen = G2Affine::generator();
    let ident = G2Affine::identity();

    let gen_p = gen.to_raw_bytes();
    let gen_p = unsafe { G2Affine::from_slice_unchecked(&gen_p) };

    let ident_p = ident.to_raw_bytes();
    let ident_p = unsafe { G2Affine::from_slice_unchecked(&ident_p) };

    assert_eq!(gen, gen_p);
    assert_eq!(ident, ident_p);
}
