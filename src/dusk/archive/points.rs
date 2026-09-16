// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

macro_rules! checked_point {
    ($native:ty, $archived:ty, [$($field:ident),+], |$point:ident| $valid:expr) => {
        #[cfg(feature = "rkyv-semantic-validation")]
        #[deny(unsafe_op_in_unsafe_fn)]
        impl<C: ?Sized> bytecheck::CheckBytes<C> for $archived {
            type Error = bytecheck::StructCheckError;

            unsafe fn check_bytes<'a>(
                value: *const Self,
                context: &mut C,
            ) -> Result<&'a Self, Self::Error> {
                $(
                    unsafe {
                        bytecheck::CheckBytes::check_bytes(&raw const (*value).$field, context)
                    }
                    .map_err(|error| bytecheck::StructCheckError {
                        field_name: stringify!($field),
                        inner: bytecheck::ErrorBox::new(error),
                    })?;
                )+

                let archived = unsafe { &*value };
                let $point: $native =
                    rkyv::Deserialize::deserialize(archived, &mut rkyv::Infallible).unwrap();
                if !bool::from($valid) {
                    return Err($crate::dusk::archive::invalid_struct(
                        "point",
                        "invalid curve, subgroup or infinity representation",
                    ));
                }

                Ok(archived)
            }
        }

        #[cfg(feature = "rkyv-validation")]
        impl $native {
            /// Decode a standalone untrusted archive with full representation,
            /// curve, subgroup and infinity checks.
            ///
            /// `rkyv-semantic-validation` applies the same semantic checks
            /// recursively when this point is nested in another archived type.
            /// Identity group elements are allowed; protocols must reject identity
            /// separately where their authorization rules require it.
            ///
            /// `bytes` must contain exactly `core::mem::size_of::<rkyv::Archived<Self>>()`
            /// bytes, aligned to `core::mem::align_of::<rkyv::Archived<Self>>()`.
            /// Use [`rkyv::AlignedVec`] for suitably aligned storage. Invalid lengths,
            /// alignment or point data return [`dusk_bytes::Error::InvalidData`].
            pub fn from_archive_bytes(bytes: &[u8]) -> Result<Self, dusk_bytes::Error> {
                if bytes.len() != core::mem::size_of::<rkyv::Archived<Self>>() {
                    return Err(dusk_bytes::Error::InvalidData);
                }
                let $point = rkyv::from_bytes::<Self>(bytes)
                    .map_err(|_| dusk_bytes::Error::InvalidData)?;

                #[cfg(feature = "rkyv-semantic-validation")]
                {
                    Ok($point)
                }

                #[cfg(not(feature = "rkyv-semantic-validation"))]
                {
                    if bool::from($valid) {
                        Ok($point)
                    } else {
                        Err(dusk_bytes::Error::InvalidData)
                    }
                }
            }
        }
    };
}

pub(crate) use checked_point;

#[cfg(all(test, feature = "rkyv-validation"))]
mod tests {
    use crate::{fp::Fp, fp2::Fp2, BlsScalar, G1Affine, G1Projective, G2Affine, G2Projective};

    macro_rules! check {
        ($type:ty, $point:expr, $valid:expr) => {{
            let bytes = rkyv::to_bytes::<_, 256>(&$point).unwrap();
            let generic_valid = if cfg!(feature = "rkyv-semantic-validation") {
                $valid
            } else {
                true
            };
            assert_eq!(rkyv::from_bytes::<$type>(&bytes).is_ok(), generic_valid);
            assert_eq!(
                rkyv::check_archived_root::<$type>(&bytes).is_ok(),
                generic_valid
            );
            let nested = rkyv::to_bytes::<_, 256>(&(17u64, std::vec![$point])).unwrap();
            assert_eq!(
                rkyv::from_bytes::<(u64, std::vec::Vec<$type>)>(&nested).is_ok(),
                generic_valid
            );
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
                        x: g.x + g.x,
                        y: g.y + g.y,
                        z: <$field>::one() + <$field>::one(),
                    },
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
                assert!(rkyv::from_bytes::<$affine>(&bytes).is_err());
                assert!($affine::from_archive_bytes(&bytes).is_err());
                let mut bytes = rkyv::to_bytes::<_, 256>(&$projective::generator()).unwrap();
                bytes.fill(0xff);
                assert!(rkyv::from_bytes::<$projective>(&bytes).is_err());
                assert!($projective::from_archive_bytes(&bytes).is_err());
            }
        };
    }
    policy!(g1_strict_archive, G1Affine, G1Projective, Fp, 48);
    policy!(g2_strict_archive, G2Affine, G2Projective, Fp2, 96);
}
