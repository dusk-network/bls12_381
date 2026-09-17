// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use core::fmt;

use serde::de::{Error, Visitor};
use serde::Deserializer;

pub(crate) fn deserialize_hex<'de, D: Deserializer<'de>, const N: usize>(
    deserializer: D,
) -> Result<[u8; N], D::Error> {
    struct Hex<const N: usize>;

    impl<'de, const N: usize> Visitor<'de> for Hex<N> {
        type Value = [u8; N];

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "a hex string encoding {N} bytes")
        }

        fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
            self.visit_bytes(value.as_bytes())
        }

        fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<Self::Value, E> {
            let mut bytes = [0; N];
            // This checks the encoded length before decoding any characters.
            hex::decode_to_slice(value, &mut bytes).map_err(E::custom)?;
            Ok(bytes)
        }
    }

    deserializer.deserialize_str(Hex::<N>)
}
