// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use dusk_bytes::{Error as BytesError, Serializable};
use subtle::ConditionallySelectable;

#[cfg(feature = "rkyv-impl")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// Wrapper for a [`subtle::Choice`]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "rkyv-impl", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Choice(u8);

impl Choice {
    pub fn unwrap_u8(&self) -> u8 {
        self.0
    }
}

impl ConditionallySelectable for Choice {
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        Self(u8::conditional_select(&a.0, &b.0, choice))
    }
}

impl Serializable<1> for Choice {
    type Error = BytesError;

    fn from_bytes(buf: &[u8; Self::SIZE]) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        if buf[0] > 1 {
            return Err(BytesError::InvalidData);
        }
        Ok(Self(buf[0]))
    }

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        [self.0; Self::SIZE]
    }
}

impl From<u8> for Choice {
    fn from(int: u8) -> Self {
        Self(int & 1)
    }
}

impl From<Choice> for u8 {
    fn from(c: Choice) -> Self {
        c.0
    }
}

impl From<subtle::Choice> for Choice {
    fn from(c: subtle::Choice) -> Self {
        Self(c.unwrap_u8())
    }
}

impl From<Choice> for subtle::Choice {
    fn from(c: Choice) -> Self {
        subtle::Choice::from(c.0)
    }
}

impl From<Choice> for bool {
    fn from(c: Choice) -> Self {
        subtle::Choice::from(c.0).into()
    }
}

#[cfg(feature = "rkyv-impl")]
const _: () = {
    use bytecheck::CheckBytes;

    impl<C: ?Sized> CheckBytes<C> for ArchivedChoice {
        type Error = bytecheck::StructCheckError;

        unsafe fn check_bytes<'a>(
            value: *const Self,
            _context: &mut C,
        ) -> Result<&'a Self, Self::Error> {
            let byte = (*value).0;
            if byte > 1 {
                return Err(bytecheck::StructCheckError {
                    field_name: "0",
                    inner: bytecheck::ErrorBox::new(bytecheck::BoolCheckError {
                        invalid_value: byte,
                    }),
                });
            }
            Ok(&*value)
        }
    }
};

#[cfg(test)]
mod tests {
    use super::*;
    use dusk_bytes::Serializable;

    #[test]
    fn from_u8_masks_input() {
        assert_eq!(Choice::from(0u8).unwrap_u8(), 0);
        assert_eq!(Choice::from(1u8).unwrap_u8(), 1);
        assert_eq!(Choice::from(2u8).unwrap_u8(), 0);
        assert_eq!(Choice::from(3u8).unwrap_u8(), 1);
        assert_eq!(Choice::from(255u8).unwrap_u8(), 1);
    }

    #[test]
    fn serializable_rejects_invalid() {
        assert!(Choice::from_bytes(&[0]).is_ok());
        assert!(Choice::from_bytes(&[1]).is_ok());
        assert!(Choice::from_bytes(&[2]).is_err());
        assert!(Choice::from_bytes(&[255]).is_err());
    }

    #[cfg(feature = "rkyv-impl")]
    mod rkyv_tests {
        use super::*;
        use bytecheck::CheckBytes;
        use rkyv::ser::serializers::AllocSerializer;
        use rkyv::ser::Serializer;
        use rkyv::{archived_root, Archived};

        #[test]
        fn rkyv_round_trip_valid() {
            for val in [0u8, 1u8] {
                let choice = Choice(val);
                let mut serializer = AllocSerializer::<256>::default();
                serializer
                    .serialize_value(&choice)
                    .expect("failed to serialize");
                let bytes = serializer.into_serializer().into_inner();
                let archived = unsafe { archived_root::<Choice>(&bytes) };
                assert_eq!(archived.0, val);
                // Validate via CheckBytes
                let ptr = archived as *const Archived<Choice>;
                let result =
                    unsafe { <Archived<Choice> as CheckBytes<()>>::check_bytes(ptr, &mut ()) };
                assert!(result.is_ok());
            }
        }

        #[test]
        fn rkyv_rejects_invalid_choice() {
            for invalid in [2u8, 128, 255] {
                let ptr = &invalid as *const u8 as *const Archived<Choice>;
                let result =
                    unsafe { <Archived<Choice> as CheckBytes<()>>::check_bytes(ptr, &mut ()) };
                assert!(
                    result.is_err(),
                    "expected rkyv validation to reject byte {invalid}"
                );
            }
        }
    }
}
