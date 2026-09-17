// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(feature = "groups")]

use dusk_bls12_381::{G1Affine, G2Affine};
use group::{prime::PrimeCurveAffine, UncompressedEncoding};

fn uncompressed_boundaries<G: PrimeCurveAffine + UncompressedEncoding>() {
    for point in [G::identity(), G::generator(), -G::generator()] {
        let bytes = point.to_uncompressed();
        assert_eq!(G::from_uncompressed(&bytes).unwrap(), point);
        assert_eq!(G::from_uncompressed_unchecked(&bytes).unwrap(), point);
        assert_eq!(G::from_bytes(&point.to_bytes()).unwrap(), point);
    }
    let reject = |bytes: &G::Uncompressed| {
        assert!(bool::from(G::from_uncompressed(bytes).is_none()));
        assert!(bool::from(G::from_uncompressed_unchecked(bytes).is_none()));
    };
    // Compression, sort, and infinity flags are invalid on this finite encoding.
    for flag in [0x80, 0x20, 0x40] {
        let mut bytes = G::generator().to_uncompressed();
        bytes.as_mut()[0] |= flag;
        reject(&bytes);
    }
    // Corrupt each Fp component independently, without setting flag bits.
    let len = G::generator().to_uncompressed().as_ref().len();
    for offset in (0..len).step_by(48) {
        let mut bytes = G::generator().to_uncompressed();
        bytes.as_mut()[offset..offset + 48].fill(0xff);
        bytes.as_mut()[offset] = 0x1f;
        reject(&bytes);
    }
    let mut infinity = G::Uncompressed::default();
    infinity.as_mut()[0] = 0x40;
    assert_eq!(G::from_uncompressed(&infinity).unwrap(), G::identity());
    for offset in [len / 2 - 1, len - 1] {
        infinity.as_mut()[offset] = 1;
        reject(&infinity);
        infinity.as_mut()[offset] = 0;
    }
    // (0, 0) has canonical coordinates but is off-curve, not infinity.
    let off_curve = G::Uncompressed::default();
    assert!(bool::from(
        G::from_uncompressed_unchecked(&off_curve).is_some()
    ));
    assert!(bool::from(G::from_uncompressed(&off_curve).is_none()));

    // Find an on-curve non-subgroup point through the separate compressed API.
    let torsion = (0..16)
        .find_map(|x| {
            let mut bytes = G::Repr::default();
            bytes.as_mut()[0] = 0x80;
            *bytes.as_mut().last_mut().unwrap() = x;
            let point = Option::<G>::from(G::from_bytes_unchecked(&bytes))?;
            bool::from(G::from_bytes(&bytes).is_none()).then_some(point)
        })
        .expect("small-x non-subgroup point");
    let bytes = torsion.to_uncompressed();
    assert_eq!(G::from_uncompressed_unchecked(&bytes).unwrap(), torsion);
    assert!(bool::from(G::from_uncompressed(&bytes).is_none()));
}

#[test]
fn g1_uncompressed_boundaries() {
    uncompressed_boundaries::<G1Affine>();
}

#[test]
fn g2_uncompressed_boundaries() {
    uncompressed_boundaries::<G2Affine>();
}

#[cfg(feature = "alloc")]
#[test]
fn variable_base_msm_matches_individual_products() {
    use dusk_bls12_381::{multiscalar_mul::msm_variable_base, BlsScalar, G1Projective};

    let points: Vec<_> = (0..65)
        .map(|i| G1Affine::from(G1Projective::generator() * BlsScalar::from(i)))
        .collect();
    let scalars: Vec<_> = (0..64)
        .map(|i| match i % 4 {
            0 => BlsScalar::zero(),
            1 => BlsScalar::one(),
            2 => -BlsScalar::one(),
            _ => BlsScalar::from(i),
        })
        .collect();
    for n in [0, 1, 2, 31, 32, 33, 63, 64] {
        let expected: G1Projective = points[..n]
            .iter()
            .zip(&scalars[..n])
            .map(|(point, scalar)| point * scalar)
            .sum();
        // An extra base must be ignored, including on the larger-window path.
        assert_eq!(msm_variable_base(&points[..n + 1], &scalars[..n]), expected);
    }
}

#[test]
fn projective_trait_roundtrips_and_batch_normalization() {
    use group::{Curve, GroupEncoding};
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    fn check<G: Curve + GroupEncoding>()
    where
        G::AffineRepr: PrimeCurveAffine<Curve = G>,
    {
        let random = G::random(XorShiftRng::from_seed([42; 16]));
        assert!(!bool::from(random.is_identity()));
        let points = [G::identity(), G::generator(), -G::generator(), random];
        let mut affine = [G::AffineRepr::identity(); 4];
        G::batch_normalize(&points, &mut affine);
        G::batch_normalize(&[], &mut []);
        for (point, affine) in points.into_iter().zip(affine) {
            assert_eq!(affine, point.to_affine());
            assert_eq!(affine.to_curve(), point);
            assert_eq!(
                bool::from(affine.is_identity()),
                bool::from(point.is_identity())
            );
            assert_eq!(G::from_bytes(&point.to_bytes()).unwrap(), point);
            assert_eq!(G::from_bytes_unchecked(&point.to_bytes()).unwrap(), point);
            assert_eq!(point.double(), point + point);
        }
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            G::batch_normalize(&points, &mut affine[..1]);
        }))
        .is_err());
        let mut invalid = G::Repr::default();
        invalid.as_mut().fill(0xff);
        assert!(bool::from(G::from_bytes(&invalid).is_none()));
        assert!(bool::from(G::from_bytes_unchecked(&invalid).is_none()));
    }
    check::<dusk_bls12_381::G1Projective>();
    check::<dusk_bls12_381::G2Projective>();
}

#[test]
fn scalar_field_traits() {
    use dusk_bls12_381::BlsScalar as F;
    use ff::{Field, PrimeField};

    for value in [F::ZERO, F::ONE, -F::ONE, F::from(7)] {
        assert_eq!(F::from_repr(value.to_repr()).unwrap(), value);
        assert_eq!(bool::from(value.is_odd()), value.to_bytes()[0] & 1 != 0);
        assert_eq!(Field::double(&value), value + value);
        let square = Field::square(&value);
        assert_eq!(square, value * value);
        assert_eq!(Field::sqrt(&square).unwrap().square(), square);
        let inverse = Field::invert(&value);
        assert_eq!(bool::from(inverse.is_some()), value != F::ZERO);
        if value != F::ZERO {
            assert_eq!(value * inverse.unwrap(), F::ONE);
        }
    }
    assert!(bool::from(F::from_repr([0xff; 32]).is_none()));
    for (num, div, square) in [
        (F::ZERO, F::ZERO, true),
        (F::ZERO, F::ONE, true),
        (F::ONE, F::ZERO, false),
        (F::from(4), F::from(9), true),
        (F::MULTIPLICATIVE_GENERATOR, F::ONE, false),
    ] {
        let (valid, root) = F::sqrt_ratio(&num, &div);
        assert_eq!(bool::from(valid), square);
        if square {
            assert_eq!(root.square() * div, num);
        }
        if num == F::ZERO || div == F::ZERO {
            assert_eq!(root, F::ZERO);
        }
    }
}
