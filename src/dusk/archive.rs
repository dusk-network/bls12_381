// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use core::fmt;

use bytecheck::{ErrorBox, TupleStructCheckError};

use crate::util::sbb;

#[derive(Debug)]
struct SemanticCheckError(&'static str);

impl fmt::Display for SemanticCheckError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl core::error::Error for SemanticCheckError {}

pub(crate) fn invalid_tuple(field_index: usize, message: &'static str) -> TupleStructCheckError {
    TupleStructCheckError {
        field_index,
        inner: ErrorBox::new(SemanticCheckError(message)),
    }
}

pub(crate) fn limbs_are_canonical<const N: usize>(limbs: &[u64; N], modulus: &[u64; N]) -> bool {
    let borrow = limbs
        .iter()
        .zip(modulus)
        .fold(0, |borrow, (&limb, &modulus)| sbb(limb, modulus, borrow).1);

    borrow != 0
}
