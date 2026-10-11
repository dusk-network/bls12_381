// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![cfg(any(feature = "hash-to-curve", feature = "experimental"))]

use dusk_bls12_381::hash_to_curve::{
    ExpandMessage, ExpandMessageState, ExpandMsgXmd, ExpandMsgXof,
};
use sha2::{Digest, Sha256};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake128;

/// The salt RFC 9380 hashes an oversized DST with.
const OVERSIZE_DST_SALT: &[u8] = b"H2C-OVERSIZE-DST-";

fn expand<X: ExpandMessage>(dst: &[u8]) -> [u8; 32] {
    let mut output = [0; 32];
    X::init_expand(b"message", dst, output.len()).read_into(&mut output);
    output
}

fn hashed_xmd_dst(dst: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, OVERSIZE_DST_SALT);
    Digest::update(&mut hasher, dst);
    hasher.finalize().to_vec()
}

fn hashed_xof_dst(dst: &[u8]) -> Vec<u8> {
    let mut hasher = Shake128::default();
    Update::update(&mut hasher, OVERSIZE_DST_SALT);
    Update::update(&mut hasher, dst);
    let mut hashed = vec![0; 32];
    hasher.finalize_xof().read(&mut hashed);
    hashed
}

#[test]
fn only_a_dst_longer_than_255_bytes_is_hashed() {
    // A 255-byte DST is used as it is, so it expands differently from its
    // hashed form, while a 256-byte DST expands as its hashed form.
    let longest_raw = [0x5a; 255];
    let shortest_hashed = [0x5a; 256];

    type Xmd = ExpandMsgXmd<Sha256>;
    assert_ne!(
        expand::<Xmd>(&longest_raw),
        expand::<Xmd>(&hashed_xmd_dst(&longest_raw))
    );
    assert_eq!(
        expand::<Xmd>(&shortest_hashed),
        expand::<Xmd>(&hashed_xmd_dst(&shortest_hashed))
    );

    type Xof = ExpandMsgXof<Shake128>;
    assert_ne!(
        expand::<Xof>(&longest_raw),
        expand::<Xof>(&hashed_xof_dst(&longest_raw))
    );
    assert_eq!(
        expand::<Xof>(&shortest_hashed),
        expand::<Xof>(&hashed_xof_dst(&shortest_hashed))
    );
}
