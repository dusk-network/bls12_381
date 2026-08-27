// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use dusk_bytes::{Error as BytesError, Serializable};

use super::G2Affine;
use crate::fp::Fp;
use crate::fp2::Fp2;

impl G2Affine {
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
    use alloc::string::{String, ToString};

    use serde::de::Error as SerdeError;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl Serialize for G2Affine {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let s = hex::encode(self.to_bytes());
            s.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for G2Affine {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let s = String::deserialize(deserializer)?;
            let decoded = hex::decode(&s).map_err(SerdeError::custom)?;
            let decoded_len = decoded.len();
            let bytes: [u8; G2Affine::SIZE] = decoded.try_into().map_err(|_| {
                SerdeError::invalid_length(decoded_len, &G2Affine::SIZE.to_string().as_str())
            })?;
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
