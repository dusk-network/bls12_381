// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use subtle::Choice;

use super::{Fp, MODULUS};
use crate::util::sbb;

#[cfg(feature = "rkyv-impl")]
use bytecheck::CheckBytes;
#[cfg(feature = "rkyv-impl")]
use rkyv::Archived;

#[cfg(feature = "rkyv-impl")]
use crate::dusk::archive::{invalid_tuple, limbs_are_canonical};

#[cfg(feature = "rkyv-impl")]
impl<C: ?Sized> CheckBytes<C> for super::ArchivedFp
where
    Archived<[u64; 6]>: CheckBytes<C>,
{
    type Error = bytecheck::TupleStructCheckError;

    unsafe fn check_bytes<'a>(
        value: *const Self,
        context: &mut C,
    ) -> Result<&'a Self, Self::Error> {
        let archived = unsafe {
            <Archived<[u64; 6]> as CheckBytes<C>>::check_bytes(&raw const (*value).0, context)
        }
        .map_err(|error| bytecheck::TupleStructCheckError {
            field_index: 0,
            inner: bytecheck::ErrorBox::new(error),
        })?;

        let limbs: [u64; 6] =
            rkyv::Deserialize::deserialize(archived, &mut rkyv::Infallible).unwrap();

        if !limbs_are_canonical(&limbs, &MODULUS) {
            return Err(invalid_tuple(0, "field element is not canonical"));
        }

        Ok(unsafe { &*value })
    }
}

impl Fp {
    /// Returns whether the internal Montgomery representation is canonical.
    pub(crate) fn is_canonical(&self) -> Choice {
        let borrow = self
            .0
            .iter()
            .zip(MODULUS)
            .fold(0, |borrow, (&limb, modulus)| sbb(limb, modulus, borrow).1);

        Choice::from((borrow as u8) & 1)
    }

    /// Internal representation of `Fp`
    pub const fn internal_repr(&self) -> &[u64; 6] {
        &self.0
    }
}

#[cfg(all(test, feature = "rkyv-impl"))]
mod rkyv_tests {
    use super::*;

    fn is_valid(value: &Fp) -> bool {
        let bytes = rkyv::to_bytes::<_, 256>(value).unwrap();
        let archived = unsafe { rkyv::archived_root::<Fp>(&bytes) };
        let ptr = archived as *const Archived<Fp>;
        unsafe { <Archived<Fp> as CheckBytes<()>>::check_bytes(ptr, &mut ()) }.is_ok()
    }

    #[test]
    fn rejects_noncanonical_field_limbs() {
        let mut largest = MODULUS;
        largest[0] -= 1;

        assert!(is_valid(&Fp::zero()));
        assert!(is_valid(&Fp(largest)));
        assert!(!is_valid(&Fp(MODULUS)));
        assert!(!is_valid(&Fp([u64::MAX; 6])));

        #[cfg(feature = "alloc")]
        {
            let bytes = rkyv::to_bytes::<_, 256>(&alloc::vec![Fp(MODULUS)]).unwrap();
            assert!(rkyv::from_bytes::<alloc::vec::Vec<Fp>>(&bytes).is_err());
        }
    }

    #[test]
    fn containing_types_reject_noncanonical_field_limbs() {
        let value = crate::fp2::Fp2 {
            c0: Fp(MODULUS),
            c1: Fp::zero(),
        };
        let bytes = rkyv::to_bytes::<_, 256>(&value).unwrap();
        let archived = unsafe { rkyv::archived_root::<crate::fp2::Fp2>(&bytes) };
        let ptr = archived as *const Archived<crate::fp2::Fp2>;

        assert!(unsafe {
            <Archived<crate::fp2::Fp2> as CheckBytes<()>>::check_bytes(ptr, &mut ())
        }
        .is_err());
    }
}

#[cfg(feature = "serde")]
mod serde_support {
    extern crate alloc;

    use alloc::string::{String, ToString};

    use serde::de::Error as SerdeError;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl Serialize for Fp {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let s = hex::encode(self.to_bytes());
            s.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Fp {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let s = String::deserialize(deserializer)?;
            let decoded = hex::decode(&s).map_err(SerdeError::custom)?;
            let decoded_len = decoded.len();
            const FP_BYTES_LEN: usize = 48;
            let bytes: [u8; FP_BYTES_LEN] = decoded.try_into().map_err(|_| {
                SerdeError::invalid_length(decoded_len, &FP_BYTES_LEN.to_string().as_str())
            })?;
            let fp = Option::from(Fp::from_bytes(&bytes))
                .ok_or(SerdeError::custom("Failed to deserialize Fp: invalid Fp"))?;
            Ok(fp)
        }
    }

    #[cfg(test)]
    mod tests {
        use alloc::boxed::Box;

        use rand::rngs::StdRng;
        use rand_core::SeedableRng;

        use super::*;
        use crate::dusk::test_utils;

        #[test]
        fn serde_fp() -> Result<(), Box<dyn std::error::Error>> {
            let mut rng = StdRng::seed_from_u64(0xc0b);
            let fp = Fp::random(&mut rng);
            let ser = test_utils::assert_canonical_json(
                &fp,
                "\"16e40954bea69030cc133b0597126df8d4d35ed26e4ed93346dcbdc306e2e92039a0d32ccd21176819a26cb9430335f2\""
            )?;
            let deser: Fp = serde_json::from_str(&ser).unwrap();
            assert_eq!(fp, deser);
            Ok(())
        }

        #[test]
        fn serde_fp_too_short_encoded() {
            let length_47_enc = "\"16e40954bea69030cc133b0597126df8d4d35ed26e4ed93346dcbdc306e2e92039a0d32ccd21176819a26cb9430335\"";

            let fp: Result<Fp, _> = serde_json::from_str(&length_47_enc);
            assert!(fp.is_err());
        }

        #[test]
        fn serde_fp_too_long_encoded() {
            let length_49_enc = "\"16e40954bea69030cc133b0597126df8d4d35ed26e4ed93346dcbdc306e2e92039a0d32ccd21176819a26cb9430335f200\"";

            let fp: Result<Fp, _> = serde_json::from_str(&length_49_enc);
            assert!(fp.is_err());
        }
    }
}
