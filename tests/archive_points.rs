// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(all(feature = "groups", feature = "rkyv-validation"))]

use core::mem::{align_of, size_of};

use dusk_bls12_381::{G1Affine, G1Projective, G2Affine, G2Projective};
use dusk_bytes::Error;
use rkyv::{AlignedVec, Archived};

macro_rules! archive_input {
    ($name:ident, $point:ty) => {
        #[test]
        fn $name() {
            let point = <$point>::generator();
            let bytes = rkyv::to_bytes::<_, 256>(&point).unwrap();
            assert_eq!(bytes.len(), size_of::<Archived<$point>>());
            assert_eq!(<$point>::from_archive_bytes(&bytes), Ok(point));

            for input in [&bytes[..0], &bytes[..1], &bytes[..bytes.len() - 1]] {
                assert_eq!(<$point>::from_archive_bytes(input), Err(Error::InvalidData));
            }

            let alignment = align_of::<Archived<$point>>();
            let mut buffer = AlignedVec::new();
            buffer.push(0xa5);
            buffer.extend_from_slice(&bytes);
            let unaligned = &buffer[1..];
            assert_eq!(unaligned, bytes.as_slice());
            assert_ne!(unaligned.as_ptr().align_offset(alignment), 0);
            assert_eq!(
                <$point>::from_archive_bytes(unaligned),
                Err(Error::InvalidData)
            );

            buffer.clear();
            buffer.resize(alignment, 0xa5);
            buffer.extend_from_slice(&bytes);
            // The root is aligned and valid: only standalone framing rejects this prefix.
            assert_eq!(rkyv::from_bytes::<$point>(&buffer).unwrap(), point);
            assert_eq!(
                <$point>::from_archive_bytes(&buffer),
                Err(Error::InvalidData)
            );
        }
    };
}

archive_input!(g1_affine_archive_input, G1Affine);
archive_input!(g1_projective_archive_input, G1Projective);
archive_input!(g2_affine_archive_input, G2Affine);
archive_input!(g2_projective_archive_input, G2Projective);
