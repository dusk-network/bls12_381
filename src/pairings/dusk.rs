// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use crate::fp::Fp;
use crate::fp2::Fp2;

use super::G2Prepared;

use alloc::vec::Vec;

impl G2Prepared {
    /// Raw bytes representation
    ///
    /// The intended usage of this function is for trusted sets of data
    /// where performance is critical. This way, the `infinity` internal
    /// attribute will not be stored and the coefficients will be stored
    /// without any check.
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        let mut bytes = alloc::vec![0u8; 288 * self.coeffs.len()];
        let mut chunks = bytes.chunks_exact_mut(8);

        self.coeffs.iter().for_each(|(a, b, c)| {
            a.c0.internal_repr()
                .iter()
                .chain(a.c1.internal_repr().iter())
                .chain(b.c0.internal_repr().iter())
                .chain(b.c1.internal_repr().iter())
                .chain(c.c0.internal_repr().iter())
                .chain(c.c1.internal_repr().iter())
                .for_each(|n| {
                    if let Some(c) = chunks.next() {
                        c.copy_from_slice(&n.to_le_bytes())
                    }
                })
        });

        bytes
    }

    /// Create a `G2Prepared` from a set of bytes created by
    /// `G2Prepared::to_raw_bytes`.
    ///
    /// # Safety
    /// No check is performed and no constant time is granted. The
    /// `infinity` attribute is also lost. The expected usage of this
    /// function is for trusted bytes where performance is critical.
    pub unsafe fn from_slice_unchecked(bytes: &[u8]) -> Self {
        let coeffs = bytes
            .chunks_exact(288)
            .map(|c| {
                let mut ac0 = [0u64; 6];
                let mut ac1 = [0u64; 6];
                let mut bc0 = [0u64; 6];
                let mut bc1 = [0u64; 6];
                let mut cc0 = [0u64; 6];
                let mut cc1 = [0u64; 6];
                let mut z = [0u8; 8];

                ac0.iter_mut()
                    .chain(ac1.iter_mut())
                    .chain(bc0.iter_mut())
                    .chain(bc1.iter_mut())
                    .chain(cc0.iter_mut())
                    .chain(cc1.iter_mut())
                    .zip(c.chunks_exact(8))
                    .for_each(|(n, c)| {
                        z.copy_from_slice(c);
                        *n = u64::from_le_bytes(z);
                    });

                let c0 = Fp::from_raw_unchecked(ac0);
                let c1 = Fp::from_raw_unchecked(ac1);
                let a = Fp2 { c0, c1 };

                let c0 = Fp::from_raw_unchecked(bc0);
                let c1 = Fp::from_raw_unchecked(bc1);
                let b = Fp2 { c0, c1 };

                let c0 = Fp::from_raw_unchecked(cc0);
                let c1 = Fp::from_raw_unchecked(cc1);
                let c = Fp2 { c0, c1 };

                (a, b, c)
            })
            .collect();
        let infinity = 0u8.into();

        Self { coeffs, infinity }
    }
}

#[cfg(feature = "serde")]
mod serde_support {
    use serde::de::{Error as SerdeError, MapAccess, Visitor};
    use serde::ser::SerializeStruct;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;
    use crate::dusk::choice::Choice;

    impl Serialize for G2Prepared {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut ser_struct = serializer.serialize_struct("G2Prepared", 2)?;
            ser_struct.serialize_field("infinity", &self.infinity.unwrap_u8())?;
            ser_struct.serialize_field("coeffs", &self.coeffs)?;
            ser_struct.end()
        }
    }

    impl<'de> Deserialize<'de> for G2Prepared {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct G2PreparedVisitor;

            const FIELDS: &[&str] = &["infinity", "coeffs"];

            impl<'de> Visitor<'de> for G2PreparedVisitor {
                type Value = G2Prepared;

                fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    formatter.write_str("a struct a with fields infinity and coeffs")
                }

                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                    let mut infinity: Option<u8> = None;
                    let mut coeffs = None;
                    while let Some(key) = map.next_key()? {
                        match key {
                            "infinity" => {
                                if infinity.is_some() {
                                    return Err(SerdeError::duplicate_field("infinity"));
                                } else {
                                    infinity = Some(map.next_value()?);
                                }
                            }
                            "coeffs" => {
                                if coeffs.is_some() {
                                    return Err(SerdeError::duplicate_field("coeffs"));
                                } else {
                                    coeffs = Some(map.next_value()?);
                                }
                            }
                            field => return Err(SerdeError::unknown_field(field, &FIELDS)),
                        }
                    }
                    Ok(G2Prepared {
                        infinity: Choice::from(
                            infinity.ok_or_else(|| SerdeError::missing_field("infinity"))?,
                        ),
                        coeffs: coeffs.ok_or_else(|| SerdeError::missing_field("coeffs"))?,
                    })
                }
            }

            deserializer.deserialize_struct("G2Prepared", FIELDS, G2PreparedVisitor)
        }
    }

    #[test]
    fn serde_g2_prepared() {
        use crate::G2Affine;

        let g2_prepared = G2Prepared::from(G2Affine::generator());
        let ser = serde_json::to_string(&g2_prepared).unwrap();
        let deser: G2Prepared = serde_json::from_str(&ser).unwrap();

        assert_eq!(g2_prepared.coeffs, deser.coeffs);
        assert_eq!(g2_prepared.infinity.unwrap_u8(), deser.infinity.unwrap_u8());
    }
}

#[test]
fn g2_prepared_bytes_unchecked() {
    use crate::G2Affine;

    let g2_prepared = G2Prepared::from(G2Affine::generator());
    let bytes = g2_prepared.to_raw_bytes();

    let g2_prepared_p = unsafe { G2Prepared::from_slice_unchecked(&bytes) };

    assert_eq!(g2_prepared.coeffs, g2_prepared_p.coeffs);
}
