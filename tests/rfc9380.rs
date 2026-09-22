// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Public-API RFC 9380 vectors; provenance and fixture license: rfc9380/README.md.
// A broken compatibility alias must fail compilation, not silently skip these tests.
#![cfg(any(feature = "hash-to-curve", feature = "experimental"))]

use dusk_bls12_381::hash_to_curve::{
    ExpandMessage, ExpandMessageState, ExpandMsgXmd, ExpandMsgXof, HashToCurve, HashToField,
    InitExpandMessage, MapToCurve,
};
use dusk_bls12_381::{G1Affine, G1Projective, G2Affine, G2Projective};
use serde_json::Value;
use sha2::{Sha256, Sha512};
use sha3::{Shake128, Shake256};

fn fixture(name: &str) -> Value {
    let mut data: Value = serde_json::from_str(include_str!("rfc9380/vectors.json")).unwrap();
    assert_eq!(data.as_object().unwrap().len(), 11);
    data[name].take()
}

fn text(value: &Value) -> &str {
    value.as_str().unwrap()
}

fn field_bytes(value: &str) -> Vec<u8> {
    value
        .split(',')
        .flat_map(|part| hex::decode(format!("{:0>96}", part.trim_start_matches("0x"))).unwrap())
        .collect()
}

fn point_bytes(value: &Value) -> Vec<u8> {
    // RFC Fp2 coordinates list c0,c1; the public G2 encoding stores c1,c0.
    ["x", "y"]
        .into_iter()
        .flat_map(|axis| text(&value[axis]).split(',').rev())
        .flat_map(field_bytes)
        .collect()
}

fn g1_field(value: &<G1Projective as MapToCurve>::Field) -> Vec<u8> {
    value.to_bytes().to_vec()
}

fn g2_field(value: &<G2Projective as MapToCurve>::Field) -> Vec<u8> {
    [value.c0.to_bytes(), value.c1.to_bytes()].concat()
}

macro_rules! curve_vectors {
    ($test:ident, $group:ty, $affine:ty, $field:ident, $file:literal) => {
        #[test]
        fn $test() {
            let data = fixture($file);
            let ro = data["randomOracle"].as_bool().unwrap();
            let vectors = data["vectors"].as_array().unwrap();
            assert_eq!(vectors.len(), 5);
            for (case, vector) in vectors.iter().enumerate() {
                let msg = text(&vector["msg"]).as_bytes();
                let dst = text(&data["dst"]).as_bytes();
                let mut fields = [<$group as MapToCurve>::Field::default(); 2];
                let count = if ro { 2 } else { 1 };
                <<$group as MapToCurve>::Field as HashToField>::hash_to_field::<
                    ExpandMsgXmd<Sha256>,
                >(msg, dst, &mut fields[..count]);
                assert_eq!(vector["u"].as_array().unwrap().len(), count);
                for (i, field) in fields[..count].iter().enumerate() {
                    assert_eq!($field(field), field_bytes(text(&vector["u"][i])), "u: {} {case}/{i}", $file);
                    let key = if ro { format!("Q{i}") } else { "Q".into() };
                    let mapped = <$affine>::from(<$group as MapToCurve>::map_to_curve(field));
                    assert!(bool::from(mapped.is_on_curve()));
                    assert_eq!(mapped.to_uncompressed().as_slice(), point_bytes(&vector[&key]), "Q: {} {case}/{i}", $file);
                }
                let output = if ro {
                    <$group as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve(msg, dst)
                } else {
                    <$group as HashToCurve<ExpandMsgXmd<Sha256>>>::encode_to_curve(msg, dst)
                };
                let output = <$affine>::from(output);
                assert!(bool::from(output.is_on_curve()));
                assert!(bool::from(output.is_torsion_free()));
                assert_eq!(output.to_uncompressed().as_slice(), point_bytes(&vector["P"]), "P: {} {case}", $file);
            }
        }
    };
}

curve_vectors!(
    rfc_g1_ro,
    G1Projective,
    G1Affine,
    g1_field,
    "BLS12381G1_XMD:SHA-256_SSWU_RO_.json"
);
curve_vectors!(
    rfc_g1_nu,
    G1Projective,
    G1Affine,
    g1_field,
    "BLS12381G1_XMD:SHA-256_SSWU_NU_.json"
);
curve_vectors!(
    rfc_g2_ro,
    G2Projective,
    G2Affine,
    g2_field,
    "BLS12381G2_XMD:SHA-256_SSWU_RO_.json"
);
curve_vectors!(
    rfc_g2_nu,
    G2Projective,
    G2Affine,
    g2_field,
    "BLS12381G2_XMD:SHA-256_SSWU_NU_.json"
);

fn check_expansion<X: ExpandMessage>(msg: &[u8], dst: &[u8], expected: &[u8]) {
    let mut direct = X::init_expand(msg, dst, expected.len());
    assert_eq!(direct.remain(), expected.len());
    assert_eq!(direct.read_into(&mut []), 0);
    assert_eq!(direct.remain(), expected.len());
    let mut oversized = vec![0xa5; expected.len() + 7];
    assert_eq!(direct.read_into(&mut oversized), expected.len());
    assert_eq!(&oversized[..expected.len()], expected);
    assert_eq!(&oversized[expected.len()..], &[0xa5; 7]);
    assert_eq!(direct.remain(), 0);
    assert_eq!(direct.read_into(&mut oversized), 0);
    let mut stream = X::init_expand(msg, dst, expected.len());
    let mut output = vec![0; expected.len()];
    for chunk in output.chunks_mut(7) {
        assert_eq!(stream.read_into(chunk), chunk.len());
    }
    assert_eq!(stream.remain(), 0);
    assert_eq!(output, expected);
}

fn expansion_file<X: ExpandMessage>(name: &str) {
    let data = fixture(name);
    let tests = data["tests"].as_array().unwrap();
    assert_eq!(tests.len(), 10);
    for vector in tests {
        let expected = hex::decode(text(&vector["uniform_bytes"])).unwrap();
        assert_eq!(
            expected.len(),
            usize::from_str_radix(text(&vector["len_in_bytes"]).trim_start_matches("0x"), 16)
                .unwrap()
        );
        check_expansion::<X>(
            text(&vector["msg"]).as_bytes(),
            text(&data["DST"]).as_bytes(),
            &expected,
        );
    }
}

#[test]
fn rfc_expansion_vectors() {
    expansion_file::<ExpandMsgXmd<Sha256>>("expand_message_xmd_SHA256_38.json");
    expansion_file::<ExpandMsgXmd<Sha256>>("expand_message_xmd_SHA256_256.json");
    expansion_file::<ExpandMsgXmd<Sha512>>("expand_message_xmd_SHA512_38.json");
    expansion_file::<ExpandMsgXof<Shake128>>("expand_message_xof_SHAKE128_36.json");
    expansion_file::<ExpandMsgXof<Shake128>>("expand_message_xof_SHAKE128_256.json");
    expansion_file::<ExpandMsgXof<Shake256>>("expand_message_xof_SHAKE256_36.json");
}

#[test]
fn xof_long_dst_retains_documented_k128() {
    let data = fixture("xof-security-parameter.json");
    let msg = hex::decode(text(&data["msg"])).unwrap();
    let dst = hex::decode(text(&data["dst"])).unwrap();
    assert!(dst.len() > 255);
    let k128 = hex::decode(text(&data["k128"])).unwrap();
    let k256 = hex::decode(text(&data["k256"])).unwrap();
    let mut output = vec![0; data["length"].as_u64().unwrap() as usize];
    assert_eq!(
        ExpandMsgXof::<Shake256>::init_expand(&msg, &dst, output.len()).read_into(&mut output),
        output.len()
    );
    assert_eq!(output, k128);
    assert_ne!(output, k256, "SHAKE256 does not implicitly select k=256");
}
