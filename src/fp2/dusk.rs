// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use super::Fp2;

#[cfg(feature = "serde")]
mod serde_support {
    use serde::de::{Error as SerdeError, MapAccess, Visitor};
    use serde::ser::SerializeStruct;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl Serialize for Fp2 {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut struct_ser = serializer.serialize_struct("Fp2", 2)?;
            struct_ser.serialize_field("c0", &self.c0)?;
            struct_ser.serialize_field("c1", &self.c1)?;
            struct_ser.end()
        }
    }

    impl<'de> Deserialize<'de> for Fp2 {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            struct Fp2Visitor;

            const FIELDS: &[&str] = &["c0", "c1"];

            impl<'de> Visitor<'de> for Fp2Visitor {
                type Value = Fp2;

                fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    formatter.write_str("a struct with fields c0 and c1")
                }

                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                    let (mut c0, mut c1) = (None, None);
                    while let Some(key) = map.next_key()? {
                        match key {
                            "c0" => {
                                if c0.is_some() {
                                    return Err(SerdeError::duplicate_field("c0"));
                                } else {
                                    c0 = Some(map.next_value()?);
                                }
                            }
                            "c1" => {
                                if c1.is_some() {
                                    return Err(SerdeError::duplicate_field("c1"));
                                } else {
                                    c1 = Some(map.next_value()?);
                                }
                            }
                            field => return Err(SerdeError::unknown_field(field, &FIELDS)),
                        }
                    }
                    Ok(Fp2 {
                        c0: c0.ok_or_else(|| SerdeError::missing_field("c0"))?,
                        c1: c1.ok_or_else(|| SerdeError::missing_field("c1"))?,
                    })
                }
            }

            deserializer.deserialize_struct("Fp2", FIELDS, Fp2Visitor)
        }
    }

    #[test]
    fn serde_fp2() {
        use rand::rngs::StdRng;
        use rand_core::SeedableRng;

        let mut rng = StdRng::seed_from_u64(0xc0b);
        let fp2 = Fp2::random(&mut rng);
        let ser = serde_json::to_string(&fp2).unwrap();
        let deser: Fp2 = serde_json::from_str(&ser).unwrap();

        assert_eq!(fp2, deser);
    }
}
