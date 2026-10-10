// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(feature = "groups")]

use dusk_bls12_381::{BlsScalar, G1Affine, G1Projective, G2Affine, G2Projective};

/// A primitive cube root of unity modulo the group order. Both curves have an
/// endomorphism `(x, y) -> (ωx, y)`, with `ω` a cube root of unity in the base
/// field, that acts on the prime-order subgroup as multiplication by such a
/// root: `[λ]P` shares its y-coordinate with `P`.
const LAMBDA: BlsScalar = BlsScalar::from_raw([0x0000_0000_ffff_ffff, 0xac45_a401_0001_a402, 0, 0]);

#[test]
fn affine_equality_compares_both_coordinates() {
    let g1 = G1Affine::generator();
    let g1_same_y = G1Affine::from(G1Projective::generator() * LAMBDA);
    let (a, b) = (g1.to_uncompressed(), g1_same_y.to_uncompressed());
    assert_eq!(a[48..], b[48..]);
    assert_ne!(a[..48], b[..48]);
    // `-g1` shares the x-coordinate instead
    for point in [-g1, g1_same_y] {
        assert_ne!(point, g1);
    }

    let g2 = G2Affine::generator();
    let g2_same_y = G2Affine::from(G2Projective::generator() * LAMBDA);
    let (a, b) = (g2.to_uncompressed(), g2_same_y.to_uncompressed());
    assert_eq!(a[96..], b[96..]);
    assert_ne!(a[..96], b[..96]);
    for point in [-g2, g2_same_y] {
        assert_ne!(point, g2);
    }
}
