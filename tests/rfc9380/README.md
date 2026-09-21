# RFC 9380 fixtures

`vectors.json` bundles the following upstream JSON files without changing their
parsed contents, from `poc/vectors/` at CFRG commit
[`664b13592116cecc9e52fb192dcde0ade36f904e`](https://github.com/cfrg/draft-irtf-cfrg-hash-to-curve/tree/664b13592116cecc9e52fb192dcde0ade36f904e/poc/vectors):

- `BLS12381G1_XMD:SHA-256_SSWU_RO_.json` (RFC 9380 J.9.1)
- `BLS12381G1_XMD:SHA-256_SSWU_NU_.json` (J.9.2)
- `BLS12381G2_XMD:SHA-256_SSWU_RO_.json` (J.10.1)
- `BLS12381G2_XMD:SHA-256_SSWU_NU_.json` (J.10.2)
- `expand_message_xmd_SHA256_38.json` (K.1)
- `expand_message_xmd_SHA256_256.json` (K.2)
- `expand_message_xmd_SHA512_38.json` (K.3)
- `expand_message_xof_SHAKE128_36.json` (K.4)
- `expand_message_xof_SHAKE128_256.json` (K.5)
- `expand_message_xof_SHAKE256_36.json` (K.6)

These 20 curve and 60 expansion cases were cross-checked against the
[final RFC text](https://www.rfc-editor.org/rfc/rfc9380.txt), including intermediate
field elements and points. Tests require no network access or external files.

`xof-security-parameter.json` is an additional, independently computed SHAKE256
long-DST case, **not a published RFC vector**. It fixes the existing `k = 128`
contract and distinguishes it from `k = 256`. Reproduce its expectations from
RFC 9380 sections 5.3.2–5.3.3 using Python's standard library:

```python
import hashlib
import json
from pathlib import Path

v = json.loads(Path("tests/rfc9380/vectors.json").read_text())["xof-security-parameter.json"]
msg, dst = bytes.fromhex(v["msg"]), bytes.fromhex(v["dst"])
assert len(dst) > 255
for k in (128, 256):
    tag = hashlib.shake_256(b"H2C-OVERSIZE-DST-" + dst).digest(2 * k // 8)
    msg_prime = msg + v["length"].to_bytes(2, "big") + tag + bytes([len(tag)])
    assert hashlib.shake_256(msg_prime).hexdigest(v["length"]) == v[f"k{k}"]
```

## Upstream fixture license

The upstream repository's [contribution policy](https://github.com/cfrg/draft-irtf-cfrg-hash-to-curve/blob/664b13592116cecc9e52fb192dcde0ade36f904e/CONTRIBUTING.md)
subjects code components to the IETF Trust's Simplified BSD License.

Copyright (c) 2023 IETF Trust and the persons identified as authors of RFC 9380.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
