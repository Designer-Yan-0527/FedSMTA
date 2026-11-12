"""Deterministic state hashing utilities shared by Server_DF / Client_DF
(BSAB pair capture, DSAR sidecar) and the diagnostics / tests that verify
them. Originally the hash half of a retired Sec.33 selector module (see
git history).

Semantics (unchanged):
  * hash_prompt_module -- order-stable state hash of a prompt module (any
    nn.Module whose state_dict is the comparison unit). Hash = sha256 over
    the sorted (key, dtype, shape, bytes) list of entries, so two states
    compare equal iff every parameter/buffer matches key-for-key,
    dtype-for-dtype, shape-for-shape and byte-for-byte.
  * hash_tensor -- sha256 over a tensor's (dtype, shape, raw bytes);
    device-agnostic (bytes are pulled from the CPU copy).
"""

import hashlib

import torch


def hash_prompt_module(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("utf-8"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def hash_tensor(t):
    h = hashlib.sha256()
    t = t.detach().cpu().contiguous()
    h.update(str(t.dtype).encode("utf-8"))
    h.update(str(tuple(t.shape)).encode("utf-8"))
    h.update(t.numpy().tobytes())
    return h.hexdigest()
