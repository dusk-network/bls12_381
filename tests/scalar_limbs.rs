// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::cmp::Ordering;

use dusk_bls12_381::BlsScalar;

/// The scalar whose Montgomery limbs are all zero but for a one in limb `k`.
///
/// Montgomery form stores `a * R` for `a`, with `R = 2^256`, so these limbs
/// hold the scalar `2^(64 k) / R`.
fn unit_limb(k: usize) -> BlsScalar {
    let r_inv = BlsScalar::pow_of_2(256).invert().unwrap();
    let scalar = BlsScalar::pow_of_2(64 * k as u64) * r_inv;

    let mut limbs = [0u64; 4];
    limbs[k] = 1;
    assert_eq!(scalar.internal_repr(), &limbs);

    scalar
}

#[test]
fn equality_compares_every_limb() {
    // Zero differs from each of these in a single limb.
    for k in 0..4 {
        assert_ne!(unit_limb(k), BlsScalar::zero());
    }
}

#[test]
fn order_compares_limbs_from_the_most_significant() {
    for k in 0..4 {
        assert_eq!(unit_limb(k).cmp(&unit_limb(k)), Ordering::Equal);
    }
    for k in 0..4 {
        // Only limb `k` differs, so a comparison that skips it sees equality.
        assert_eq!(unit_limb(k).cmp(&BlsScalar::zero()), Ordering::Greater);
        assert_eq!(BlsScalar::zero().cmp(&unit_limb(k)), Ordering::Less);
    }
    for k in 1..4 {
        assert_eq!(unit_limb(k).cmp(&unit_limb(k - 1)), Ordering::Greater);
        assert_eq!(unit_limb(k - 1).cmp(&unit_limb(k)), Ordering::Less);
    }
}
