"""PSC (--psc) CPU-only unit tests: consolidation math + bookkeeping.

No data / GPU / ViT needed: Server_DF and Client_DF are created via
__new__ (no __init__), and the surrogate measurement pass is replaced
with a FIXED synthetic query bank, mirroring production semantics (the
surrogate queries are fixed data; only the key moves), so the tests
exercise exactly the consolidate_prompts() decision logic:

  T1  streak + merge commit: a pair whose routing-neighborhood centroids
      stay similar reaches streak >= K at the task-boundary fusion round
      and commits — row i := average(i, j) (CORE), row j unchanged.
  T2  drift rollback: when the merge moves surrogate top-1 destinations
      out of the pair beyond --psc_drift, the merge is rolled back from
      the pre-merge snapshot (row i restored, no core/streak change).
  T3  merge cap + row exclusion: pairs sharing a row with a committed
      merge are skipped and --psc_max_merge is honored.
  T4  streak reset: a pair that cannot be measured in a round (row's
      neighborhood below the minimum) loses its streak ("consecutive"
      semantics).
  T5  client freeze: _psc_post_step_restore restores exactly the
      residual rows (prompt_key + prompt content), leaving other rows
      untouched; _psc_prompt_top1 routes by l2-normalized cosine top-1;
      a stale-task freeze is a no-op.
  T6  merge-only invariant: after a committed merge the residual row j
      keeps its value verbatim (no deletion).

Usage (CPU, seconds):
  python tests/test_psc_consolidation.py
"""

import math
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Models.Client_DF import Client_DF  # noqa: E402
from Models.Server_DF import Server_DF  # noqa: E402

PSC_ARGS = {
    "psc": True, "psc_tau": 0.95, "psc_streak": 3,
    "psc_max_merge": 2, "psc_drift": 0.05,
}
ANGLE = 10.0          # pair angle: cos(10 deg) = 0.9848 >= tau
MIN_NEIGHBORHOOD = 2  # must mirror consolidate_prompts' constant


class FakePrompt(nn.Module):
    """Minimal Global_Prompt stand-in: prompt_key (P, C) + prompt
    (P, length, C) parameters — the only attributes consolidate_prompts
    and the client freeze touch."""

    def __init__(self, pool, dim, length=2, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.prompt_key = nn.Parameter(
            torch.randn(pool, dim, generator=gen))
        self.prompt = nn.Parameter(
            torch.randn(pool, length, dim, generator=gen))


def unit_dir(dim, theta_deg):
    v = torch.zeros(dim)
    v[0] = math.cos(math.radians(theta_deg))
    v[1] = math.sin(math.radians(theta_deg))
    return v


class QueryBank:
    """Fixed surrogate queries (production: the surrogate data never
    changes; only the key moves). Queries for row r sit on the row's
    ORIGINAL base direction + tiny deterministic jitter, so top-1
    assignment is controlled and stable across merges."""

    def __init__(self, pool, dim, max_n=6, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.pool = pool
        self.max_n = max_n
        self.base = [torch.randn(dim, generator=gen)
                     / torch.randn(1, generator=gen).norm()
                     for _ in range(pool)]
        # a fixed jitter direction per row (orthogonal-ish noise)
        self.jitter = [torch.randn(dim, generator=gen) * 0.01
                       for _ in range(pool)]

    def set_row_direction(self, row, direction):
        self.base[row] = direction / direction.norm()

    def build(self, n_per_row, dim=None):
        dim = dim or self.dim
        feats = []
        for row, n in enumerate(n_per_row):
            n = min(n, self.max_n)
            if n <= 0:
                continue
            b = self.base[row]
            j = self.jitter[row] / self.jitter[row].norm()
            for k in range(1, n + 1):
                feats.append(b + 0.01 * k * j)
        return torch.stack(feats) if feats else torch.zeros(0, dim)


def make_server(psc_args, class_mask=None, task_id=0, global_epoch=5):
    server = Server_DF.__new__(Server_DF)
    server.args = Namespace(**psc_args)
    server.global_epoch = global_epoch
    server.task_id = task_id
    server.device = "cpu"
    server.class_mask = class_mask or [[[0, 1], [2, 3]]]
    server.psc_on = True
    server._psc_core_slots = []
    server._psc_pair_streak = {}
    server._psc_merged_pairs = []
    return server


def synthetic_measurement(prompt, bank, n_per_row):
    """Same dict _psc_surrogate_routing builds, from the fixed query bank
    (queries NEVER move — only the key does, as in production)."""
    cf = bank.build(n_per_row)
    key = prompt.prompt_key.detach()

    def l2n(t):
        return t / (t.norm(dim=1, keepdim=True) + 1e-12)

    x_norm = l2n(cf)
    key_norm = l2n(key)
    sim = x_norm @ key_norm.t()
    top1 = torch.argmax(sim, dim=1)
    P = key.shape[0]
    counts = torch.zeros(P, dtype=torch.long)
    counts.scatter_add_(0, top1, torch.ones_like(top1))
    centroids = torch.zeros(P, cf.shape[1])
    centroids.index_add_(0, top1, cf)
    nonempty = counts > 0
    centroids[nonempty] = (
        centroids[nonempty] / counts[nonempty].unsqueeze(1).float())
    return {"x_norm": x_norm, "top1": top1, "counts": counts,
            "centroids": centroids, "n_samples": int(top1.numel())}


def run_rounds(server, prompt, bank, n_per_row, rounds):
    original = Server_DF._psc_surrogate_routing

    def patched(self, _p=prompt, _b=bank, _n=n_per_row):
        return synthetic_measurement(_p, _b, _n)

    Server_DF._psc_surrogate_routing = patched
    try:
        for r in rounds:
            server.prompt = prompt
            server.consolidate_prompts(r)
    finally:
        Server_DF._psc_surrogate_routing = original


def test_t1_streak_and_commit():
    prompt = FakePrompt(6, 8, seed=1)
    # pair rows at a 10-deg angle: cos = 0.9848 >= tau, but distinct
    # neighborhoods (each row's queries self-route to that row)
    prompt.prompt_key.data[0] = unit_dir(8, 0.0)
    prompt.prompt_key.data[1] = unit_dir(8, ANGLE)
    bank = QueryBank(6, 8, seed=11)
    bank.set_row_direction(0, unit_dir(8, 0.0))
    bank.set_row_direction(1, unit_dir(8, ANGLE))
    server = make_server(dict(PSC_ARGS))
    n = [4, 4, 2, 2, 2, 2]
    run_rounds(server, prompt, bank, n, rounds=[0, 1, 2])
    assert server._psc_pair_streak.get("0|1", 0) == 3, (
        "pair (0,1) must reach streak 3 after 3 similar fusion rounds")
    pre_key = prompt.prompt_key.data.clone()
    pre_prompt = prompt.prompt.data.clone()
    # round 3 = task-boundary fusion (ge=5): merge must commit
    run_rounds(server, prompt, bank, n, rounds=[3])
    expected_key = 0.5 * (pre_key[0] + pre_key[1])
    expected_prompt = 0.5 * (pre_prompt[0] + pre_prompt[1])
    assert server._psc_core_slots == [0], (
        f"row 0 must be the committed core row, got "
        f"{server._psc_core_slots}")
    assert "0|1" in server._psc_merged_pairs
    assert torch.allclose(prompt.prompt_key.data[0], expected_key), (
        "core row key must be the exact average of the two rows")
    assert torch.allclose(prompt.prompt.data[0], expected_prompt), (
        "core row prompt content must be the exact average")
    # merged pair must never re-trigger
    run_rounds(server, prompt, bank, n, rounds=[4 + 0])  # round 4: no fusion? (skipped by harness; just streak path)
    assert "0|1" not in server._psc_pair_streak, (
        "a merged pair must not accumulate a new streak")
    print("[PASS] T1 streak + merge commit")


def test_t2_drift_rollback():
    prompt = FakePrompt(4, 8, seed=2)
    prompt.prompt_key.data[0] = unit_dir(8, 0.0)
    prompt.prompt_key.data[1] = unit_dir(8, ANGLE)
    # row 2 sits almost exactly on row 0's original direction: pre-merge it
    # steals nothing, but after the merge row 0 moves to the bisector and
    # row 2 becomes strictly closer to row 0's old queries -> they flip OUT
    # of the pair (drift >> gate)
    prompt.prompt_key.data[2] = (0.98 * unit_dir(8, 0.0)
                                 + 0.02 * unit_dir(8, ANGLE))
    bank = QueryBank(4, 8, seed=12)
    bank.set_row_direction(0, unit_dir(8, 0.0))
    bank.set_row_direction(1, unit_dir(8, ANGLE))
    bank.set_row_direction(2, (0.98 * unit_dir(8, 0.0)
                               + 0.02 * unit_dir(8, ANGLE)))
    server = make_server(dict(PSC_ARGS))
    n = [6, 6, 2, 0]
    run_rounds(server, prompt, bank, n, rounds=[0, 1, 2])
    assert server._psc_pair_streak.get("0|1", 0) == 3
    snap_key = prompt.prompt_key.data.clone()
    snap_prompt = prompt.prompt.data.clone()
    run_rounds(server, prompt, bank, n, rounds=[3])  # boundary
    assert server._psc_core_slots == [], (
        "a drifting merge must NOT commit core rows")
    assert torch.allclose(prompt.prompt_key.data, snap_key) and \
        torch.allclose(prompt.prompt.data, snap_prompt), (
        "rolled-back merge must restore row 0 from the pre-merge snapshot")
    assert "0|1" not in server._psc_merged_pairs
    print("[PASS] T2 drift rollback")


def test_t3_cap_and_row_exclusion():
    prompt = FakePrompt(8, 8, seed=3)
    # two similar pairs sharing row 1: (0,1) at 10 deg, (1,2) at 12 deg
    # -> (0,1) has the higher similarity and commits first; (1,2) must be
    # excluded by the shared-row rule, and max_merge=1 caps the round
    prompt.prompt_key.data[0] = unit_dir(8, 0.0)
    prompt.prompt_key.data[1] = unit_dir(8, ANGLE)
    prompt.prompt_key.data[2] = unit_dir(8, ANGLE + 12.0)
    bank = QueryBank(8, 8, seed=13)
    bank.set_row_direction(0, unit_dir(8, 0.0))
    bank.set_row_direction(1, unit_dir(8, ANGLE))
    bank.set_row_direction(2, unit_dir(8, ANGLE + 12.0))
    server = make_server({**PSC_ARGS, "psc_max_merge": 1})
    n = [3, 3, 3, 0, 0, 0, 0, 0]
    run_rounds(server, prompt, bank, n, rounds=[0, 1, 2, 3])
    assert server._psc_core_slots == [0], (
        "with max_merge=1 only the best pair commits; the pair sharing "
        "row 1 must be excluded")
    assert server._psc_merged_pairs == ["0|1"]
    print("[PASS] T3 merge cap + shared-row exclusion")


def test_t4_streak_reset_on_unmeasurable():
    prompt = FakePrompt(6, 8, seed=4)
    prompt.prompt_key.data[0] = unit_dir(8, 0.0)
    prompt.prompt_key.data[1] = unit_dir(8, ANGLE)
    bank = QueryBank(6, 8, seed=14)
    bank.set_row_direction(0, unit_dir(8, 0.0))
    bank.set_row_direction(1, unit_dir(8, ANGLE))
    server = make_server(dict(PSC_ARGS))
    # rounds 0-1: both neighborhoods measurable, streak -> 2
    run_rounds(server, prompt, bank, [4, 4, 0, 0, 0, 0], rounds=[0, 1])
    assert server._psc_pair_streak.get("0|1", 0) == 2
    # round 2: row 1's neighborhood shrinks below the minimum -> the pair
    # is no longer measurable -> the streak MUST reset
    run_rounds(server, prompt, bank, [4, 1, 0, 0, 0, 0], rounds=[2])
    assert "0|1" not in server._psc_pair_streak, (
        "unmeasured pairs must lose their streak (consecutive semantics)")
    print("[PASS] T4 streak reset")


def test_t5_client_freeze_and_top1():
    client = Client_DF.__new__(Client_DF)
    client.device = "cpu"
    client.task_id = 1
    client._psc_on = True
    client._psc_core_slots = set()
    vit_prompts = FakePrompt(6, 8, seed=5)

    class FakeVit:
        def get_prompts(self):
            return vit_prompts

    client.vit = FakeVit()
    rows = torch.tensor([2, 5])
    client._psc_residual_slots = rows
    client._psc_key_snapshot = vit_prompts.prompt_key.data[rows].clone()
    client._psc_prompt_snapshot = vit_prompts.prompt.data[rows].clone()
    client._psc_task_id = 1

    before_key = vit_prompts.prompt_key.data.clone()
    before_prompt = vit_prompts.prompt.data.clone()
    with torch.no_grad():  # simulate optimizer steps: perturb EVERYTHING
        vit_prompts.prompt_key.data += torch.randn_like(
            vit_prompts.prompt_key.data)
        vit_prompts.prompt.data += torch.randn_like(
            vit_prompts.prompt.data)
    client._psc_post_step_restore()
    assert torch.allclose(vit_prompts.prompt_key.data[rows],
                          before_key[rows]), "residual key rows restored"
    assert torch.allclose(vit_prompts.prompt.data[rows],
                          before_prompt[rows]), \
        "residual prompt rows restored"
    assert not torch.allclose(vit_prompts.prompt_key.data, before_key), (
        "non-residual rows must keep the optimizer's update")

    # _psc_prompt_top1: normalized-cosine top-1 sanity (scale invariance)
    key = vit_prompts.prompt_key.detach()
    feats = torch.stack([key[3] * 2.0, key[3] * 0.5])
    top1 = client._psc_prompt_top1(feats, key)
    assert top1.tolist() == [3, 3], (
        f"expected both queries to route to row 3, got {top1.tolist()}")

    # stale-task guard: _psc_task_id mismatch -> restore is a no-op
    client._psc_task_id = 0
    mutated = vit_prompts.prompt_key.data.clone()
    with torch.no_grad():
        vit_prompts.prompt_key.data[rows] += 1.0
    client._psc_post_step_restore()
    assert torch.allclose(vit_prompts.prompt_key.data[rows],
                          mutated[rows] + 1.0), (
        "stale freeze (task mismatch) must be a no-op")
    print("[PASS] T5 client freeze + top-1 routing")


def test_t6_merge_only_invariant():
    prompt = FakePrompt(4, 8, seed=6)
    prompt.prompt_key.data[0] = unit_dir(8, 0.0)
    prompt.prompt_key.data[1] = unit_dir(8, ANGLE)
    bank = QueryBank(4, 8, seed=15)
    bank.set_row_direction(0, unit_dir(8, 0.0))
    bank.set_row_direction(1, unit_dir(8, ANGLE))
    server = make_server(dict(PSC_ARGS))
    run_rounds(server, prompt, bank, [4, 4, 0, 0],
               rounds=[0, 1, 2, 3])
    j_before = prompt.prompt_key.data[1].clone()
    pj_before = prompt.prompt.data[1].clone()
    assert torch.allclose(prompt.prompt_key.data[1], j_before), (
        "the residual row must keep its key verbatim (no deletion)")
    assert torch.allclose(prompt.prompt.data[1], pj_before), (
        "the residual row must keep its prompt content verbatim")
    # span invariant: {avg, row_j} spans the same space as {row_i, row_j}
    avg = prompt.prompt_key.data[0]
    d = j_before - avg
    assert d.norm() > 1e-6, (
        "the differential direction must survive in the residual row")
    print("[PASS] T6 merge-only invariant")


def main():
    test_t1_streak_and_commit()
    test_t2_drift_rollback()
    test_t3_cap_and_row_exclusion()
    test_t4_streak_reset_on_unmeasurable()
    test_t5_client_freeze_and_top1()
    test_t6_merge_only_invariant()
    print("\nRESULT: PASS (all PSC consolidation unit tests)")


if __name__ == "__main__":
    main()
