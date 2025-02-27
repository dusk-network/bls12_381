// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use super::Fp6;

#[cfg(feature = "serde")]
mod serde_support {
    use serde::de::{Error as SerdeError, MapAccess, Visitor};
    use serde::ser::SerializeStruct;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl Serialize for Fp6 {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut ser_struct = serializer.serialize_struct("Fp6", 3)?;
            ser_struct.serialize_field("c0", &self.c0)?;
            ser_struct.serialize_field("c1", &self.c1)?;
            ser_struct.serialize_field("c2", &self.c2)?;
            ser_struct.end()
        }
    }

    impl<'de> Deserialize<'de> for Fp6 {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct Fp6Visitor;

            const FIELDS: &[&str] = &["c0", "c1", "c2"];

            impl<'de> Visitor<'de> for Fp6Visitor {
                type Value = Fp6;

                fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    formatter.write_str("a struct with fields c0, c1 and c2")
                }

                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                    let (mut c0, mut c1, mut c2) = (None, None, None);
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
                            "c2" => {
                                if c2.is_some() {
                                    return Err(SerdeError::duplicate_field("c2"));
                                } else {
                                    c2 = Some(map.next_value()?);
                                }
                            }
                            field => return Err(SerdeError::unknown_field(field, &FIELDS)),
                        }
                    }
                    Ok(Fp6 {
                        c0: c0.ok_or_else(|| SerdeError::missing_field("c0"))?,
                        c1: c1.ok_or_else(|| SerdeError::missing_field("c1"))?,
                        c2: c2.ok_or_else(|| SerdeError::missing_field("c2"))?,
                    })
                }
            }

            deserializer.deserialize_struct("Fp6", FIELDS, Fp6Visitor)
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
        fn serde_fp6() -> Result<(), Box<dyn std::error::Error>> {
            let mut rng = StdRng::seed_from_u64(0xc0b);
            let fp6 = Fp6::random(&mut rng);
            let ser = test_utils::assert_canonical_json(&fp6, include_str!("./fp6.json"))?;
            let deser: Fp6 = serde_json::from_str(&ser).unwrap();

            assert_eq!(fp6, deser);
            Ok(())
        }
    }
}
