// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use dusk_bytes::{Error as BytesError, Serializable};

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

    /// Create a `G1Affine` from a set of bytes created by `G1Affine::to_raw_bytes`.
    ///
    /// # Safety
    /// No check is performed and no constant time is granted. The expected
    /// usage of this function is for trusted bytes where performance is critical.
    /// For secure serialization, check `from_bytes`.
    /// After generating the point, you can check `is_on_curve` and
    /// `is_torsion_free` to grant its security.
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
