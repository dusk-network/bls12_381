// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(feature = "pairings")]

use dusk_bls12_381::{pairing, G1Affine, G2Affine, Gt};

#[test]
fn pairing_of_two_identities_is_the_identity() {
    // Each argument is tested for the identity on its own, which must also
    // hold when both are the identity.
    for (p, q) in [
        (G1Affine::identity(), G2Affine::identity()),
        (G1Affine::generator(), G2Affine::identity()),
        (G1Affine::identity(), G2Affine::generator()),
    ] {
        assert_eq!(pairing(&p, &q), Gt::identity());
    }
}
