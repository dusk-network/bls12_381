// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use dusk_bytes::Error;
use subtle::ConstantTimeEq;

use crate::{fp::Fp, fp2::Fp2, G1Affine, G1Projective, G2Affine, G2Projective};

macro_rules! checked_archive {
    ($affine:ty, $projective:ty, $field:ty) => {
        impl $affine {
            /// Decode an untrusted archive, checking representation, curve,
            /// subgroup and infinity coordinates. Requires `rkyv-validation`.
            ///
            /// Identity is allowed, but its affine coordinates must be (0, 1).
            /// Protocols using public keys must additionally reject identity.
            /// Existing `CheckBytes` and archive layouts remain unchanged;
            /// trusted structural-only decoding does not call this method.
            /// Invalid archives return [`dusk_bytes::Error::InvalidData`].
            pub fn from_archive_bytes(bytes: &[u8]) -> Result<Self, Error> {
                let point = rkyv::from_bytes::<Self>(bytes).map_err(|_| Error::InvalidData)?;
                let infinity =
                    !point.is_identity() | (point.x.is_zero() & point.y.ct_eq(&<$field>::one()));
                if bool::from(infinity & point.is_on_curve() & point.is_torsion_free()) {
                    Ok(point)
                } else {
                    Err(Error::InvalidData)
                }
            }
        }

        impl $projective {
            /// Decode an untrusted archive with full prime-order point checks.
            /// Requires `rkyv-validation`; returns `InvalidData` on failure.
            ///
            /// Homogeneous infinity (0 : nonzero : 0) is allowed, including
            /// legitimate rescalings. Coordinates are not normalized on return.
            /// Existing `CheckBytes` and archive layouts remain unchanged.
            pub fn from_archive_bytes(bytes: &[u8]) -> Result<Self, Error> {
                let point = rkyv::from_bytes::<Self>(bytes).map_err(|_| Error::InvalidData)?;
                let infinity = !point.is_identity() | (point.x.is_zero() & !point.y.is_zero());
                if bool::from(
                    infinity & point.is_on_curve() & <$affine>::from(point).is_torsion_free(),
                ) {
                    Ok(point)
                } else {
                    Err(Error::InvalidData)
                }
            }
        }
    };
}

checked_archive!(G1Affine, G1Projective, Fp);
checked_archive!(G2Affine, G2Projective, Fp2);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlsScalar;

    macro_rules! check {
        ($type:ty, $point:expr, $valid:expr) => {{
            let bytes = rkyv::to_bytes::<_, 256>(&$point).unwrap();
            // These fixtures must reach the semantic layer, not fail layout.
            assert!(rkyv::from_bytes::<$type>(&bytes).is_ok());
            let result = <$type>::from_archive_bytes(&bytes);
            assert_eq!(result.is_ok(), $valid);
            if let Ok(point) = result {
                assert_eq!(
                    bytes.as_slice(),
                    rkyv::to_bytes::<_, 256>(&point).unwrap().as_slice()
                );
            }
            assert!(<$type>::from_archive_bytes(&bytes[..1]).is_err());
        }};
    }

    macro_rules! policy {
        ($name:ident, $affine:ident, $projective:ident, $field:ty, $size:expr) => {
            #[test]
            fn $name() {
                let g = $affine::generator();
                for point in [g, $affine::identity(), (g * BlsScalar::from(37u64)).into()] {
                    check!($affine, point, true);
                }
                for point in [
                    $projective::generator(),
                    $projective::identity(),
                    g * BlsScalar::from(37u64),
                    $projective {
                        x: <$field>::zero(),
                        y: <$field>::one() + <$field>::one(),
                        z: <$field>::zero(),
                    },
                ] {
                    check!($projective, point, true);
                }
                let torsion = (0..=u8::MAX)
                    .find_map(|x| {
                        let mut bytes = [0u8; $size];
                        bytes[0] = 0x80;
                        bytes[$size - 1] = x;
                        let point =
                            Option::<$affine>::from($affine::from_compressed_unchecked(&bytes))?;
                        (!bool::from(point.is_torsion_free())).then_some(point)
                    })
                    .unwrap();
                assert!(bool::from(torsion.is_on_curve()));
                // SAFETY: exact-size buffers, canonical limbs and boolean flags;
                // deliberately invalid curve semantics are the test input.
                let off_curve = unsafe { $affine::from_slice_unchecked(&[0; $affine::RAW_SIZE]) };
                let mut raw = g.to_raw_bytes();
                raw[$affine::RAW_SIZE - 1] = 1;
                let infinity_alias = unsafe { $affine::from_slice_unchecked(&raw) };
                for point in [torsion, off_curve, infinity_alias] {
                    check!($affine, point, false);
                }
                for point in [
                    $projective::from(torsion),
                    $projective::from(off_curve),
                    $projective {
                        x: <$field>::one(),
                        y: <$field>::one(),
                        z: <$field>::zero(),
                    },
                    $projective {
                        x: <$field>::zero(),
                        y: <$field>::zero(),
                        z: <$field>::zero(),
                    },
                ] {
                    check!($projective, point, false);
                }
                let mut bytes = rkyv::to_bytes::<_, 256>(&g).unwrap();
                bytes.fill(0xff);
                assert!($affine::from_archive_bytes(&bytes).is_err());
            }
        };
    }
    policy!(g1_strict_archive, G1Affine, G1Projective, Fp, 48);
    policy!(g2_strict_archive, G2Affine, G2Projective, Fp2, 96);
}
