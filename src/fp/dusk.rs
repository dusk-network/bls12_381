// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use super::Fp;

impl Fp {
    /// Internal representation of `Fp`
    pub const fn internal_repr(&self) -> &[u64; 6] {
        &self.0
    }
}

#[cfg(feature = "serde")]
mod serde_support {
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
            let fp = Fp::from_bytes(&bytes)
                .into_option()
                .ok_or(SerdeError::custom("Failed to deserialize Fp: invalid Fp"))?;
            Ok(fp)
        }
    }

    #[test]
    fn serde_fp() {
        use rand::rngs::StdRng;
        use rand_core::SeedableRng;

        let mut rng = StdRng::seed_from_u64(0xc0b);
        let fp = Fp::random(&mut rng);
        let ser = serde_json::to_string(&fp).unwrap();
        let deser: Fp = serde_json::from_str(&ser).unwrap();
        assert_eq!(fp, deser);
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
