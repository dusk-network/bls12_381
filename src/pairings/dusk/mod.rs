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
        let mut chunks = bytes.as_chunks_mut::<8>().0.iter_mut();

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
            .as_chunks::<288>()
            .0
            .iter()
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
                    .zip(c.as_chunks::<8>().0.iter())
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
    use serde::ser::SerializeStruct;
    use serde::{Serialize, Serializer};

    use super::*;

    impl Serialize for G2Prepared {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut ser_struct = serializer.serialize_struct("G2Prepared", 2)?;
            ser_struct.serialize_field("infinity", &self.infinity.unwrap_u8())?;
            ser_struct.serialize_field("coeffs", &self.coeffs)?;
            ser_struct.end()
        }
    }

    #[cfg(test)]
    mod tests {
        use alloc::boxed::Box;

        use super::*;
        use crate::dusk::test_utils;
        use crate::G2Affine;

        #[test]
        fn serializes_g2_prepared_canonically() -> Result<(), Box<dyn std::error::Error>> {
            let g2_prepared = G2Prepared::from(G2Affine::generator());
            test_utils::assert_canonical_json(&g2_prepared, include_str!("./g2_prepared.json"))?;
            Ok(())
        }
    }
}

#[cfg(all(feature = "rkyv-impl", feature = "alloc"))]
#[test]
fn trusted_pairing_archives_still_round_trip() {
    use crate::{multi_miller_loop, G1Affine, G2Affine, Gt, MillerLoopResult};
    use rkyv::Deserialize;

    let prepared = G2Prepared::from(G2Affine::generator());
    let bytes = rkyv::to_bytes::<_, 1024>(&prepared).unwrap();
    let archived = unsafe { rkyv::archived_root::<G2Prepared>(&bytes) };
    let restored: G2Prepared = archived.deserialize(&mut rkyv::Infallible).unwrap();
    assert_eq!(prepared.infinity.unwrap_u8(), restored.infinity.unwrap_u8());
    assert_eq!(prepared.coeffs, restored.coeffs);

    let miller = multi_miller_loop(&[(&G1Affine::generator(), &prepared)]);
    let bytes = rkyv::to_bytes::<_, 1024>(&miller).unwrap();
    let archived = unsafe { rkyv::archived_root::<MillerLoopResult>(&bytes) };
    let restored: MillerLoopResult = archived.deserialize(&mut rkyv::Infallible).unwrap();
    assert_eq!(miller.0, restored.0);

    let target = miller.final_exponentiation();
    let bytes = rkyv::to_bytes::<_, 1024>(&target).unwrap();
    let archived = unsafe { rkyv::archived_root::<Gt>(&bytes) };
    let restored: Gt = archived.deserialize(&mut rkyv::Infallible).unwrap();
    assert_eq!(target, restored);
}

#[test]
fn g2_prepared_bytes_unchecked() {
    use crate::G2Affine;

    let g2_prepared = G2Prepared::from(G2Affine::generator());
    let bytes = g2_prepared.to_raw_bytes();

    let g2_prepared_p = unsafe { G2Prepared::from_slice_unchecked(&bytes) };

    assert_eq!(g2_prepared.coeffs, g2_prepared_p.coeffs);
}

/// Compressed order-13 point on the G2 curve, outside the prime-order subgroup.
#[cfg(test)]
const ORDER_13_G2: &str = "b155337267dcdc648fb817356e8e9e26e0c729e5543bc72a0424741f956d341eb560f91e5c8ff5ed896391e6a2e9b028109965b41eafc380c9cafca101976929bd74ecdb031d1dd345e76d904fd528d70f882714e49a4a10efefb2aadfd0b1ba";

#[test]
fn zero_miller_loop_result_fails_closed() {
    use crate::fp12::Fp12;
    use crate::{Gt, MillerLoopResult};

    let gt = MillerLoopResult(Fp12::zero()).final_exponentiation();
    assert_eq!(gt.0, Fp12::zero());
    assert_ne!(gt, Gt::identity());

    // Zero is not a group element and never equals any value, even zero.
    let other = MillerLoopResult(Fp12::zero()).final_exponentiation();
    assert_ne!(gt, other);
}

#[test]
fn order_13_g2_point_fails_closed() {
    use crate::fp12::Fp12;
    use crate::{multi_miller_loop, pairing, BlsScalar, G1Affine, G2Affine, G2Projective, Gt};

    let bytes: [u8; 96] = hex::decode(ORDER_13_G2).unwrap().try_into().unwrap();
    assert!(bool::from(G2Affine::from_compressed(&bytes).is_none()));
    let q = G2Affine::from_compressed_unchecked(&bytes).unwrap();
    assert!(!bool::from(q.is_identity()));
    assert_eq!(
        G2Projective::from(q) * BlsScalar::from(13u64),
        G2Projective::identity()
    );

    let p = G1Affine::generator();
    let gt = pairing(&p, &q);
    assert_eq!(gt.0, Fp12::zero());
    assert_ne!(gt, Gt::identity());

    // Two failed pairings with different G1 inputs must not compare equal.
    let p2 = G1Affine::from(G1Affine::generator() * BlsScalar::from(2u64));
    assert_ne!(gt, pairing(&p2, &q));

    // A verification-style check e(p, g2) * e(-p, q) == 1 must fail, not panic.
    let check = multi_miller_loop(&[
        (&p, &G2Prepared::from(G2Affine::generator())),
        (&-p, &G2Prepared::from(q)),
    ])
    .final_exponentiation();
    assert_ne!(check, Gt::identity());
}
