// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(all(feature = "groups", feature = "rkyv-validation"))]

use dusk_bls12_381::{G1Affine, G2Affine};
use dusk_bytes::Error;

macro_rules! identity_aliases {
    ($name:ident, $affine:ty, $coordinate:expr) => {
        #[test]
        fn $name() {
            // The identity is `(0, 1)` under the infinity flag. The flag with
            // any other coordinates, such as `(0, 0)` or `(1, 1)`, is an alias
            // the archive checks must reject.
            let canonical = <$affine>::identity().to_raw_bytes();
            let coordinate = $coordinate;
            let mut zero_y = canonical;
            zero_y[coordinate..2 * coordinate].fill(0);
            let mut one_x = canonical;
            one_x.copy_within(coordinate..2 * coordinate, 0);

            for (raw, valid) in [(canonical, true), (zero_y, false), (one_x, false)] {
                // SAFETY: exact-size buffer, canonical limbs and a boolean flag;
                // the identity aliases are the test input.
                let point = unsafe { <$affine>::from_slice_unchecked(&raw) };
                let bytes = rkyv::to_bytes::<_, 256>(&point).unwrap();
                let decoded = <$affine>::from_archive_bytes(&bytes);
                if valid {
                    assert_eq!(decoded, Ok(<$affine>::identity()));
                } else {
                    assert_eq!(decoded, Err(Error::InvalidData));
                }
            }
        }
    };
}

identity_aliases!(g1_archive_rejects_identity_aliases, G1Affine, 48);
identity_aliases!(g2_archive_rejects_identity_aliases, G2Affine, 96);
