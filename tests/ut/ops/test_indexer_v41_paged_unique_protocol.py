# SPDX-License-Identifier: Apache-2.0
"""Randomized schedules for the two-slot AIC / metadata-AIV / topk-AIV protocol.

This checks token counts and slot ownership, not hardware memory visibility.
Native graph replay and changed-content oracles remain required.
"""

import random

import pytest


def simulate(rows: int, seed: int) -> None:
    rng = random.Random(seed)
    streams = {"metadata": [], "cube": [], "topk": []}
    for row in range(rows):
        if row >= 2:
            streams["metadata"].append(("wait", "free", row))
        streams["metadata"].extend(
            [("metadata_begin", "", row), ("metadata_end", "", row), ("set", "ready", row), ("set", "ack", row)]
        )
        if row >= 1:
            streams["metadata"].append(("wait", "scored", row - 1))
    if rows:
        streams["metadata"].append(("wait", "scored", rows - 1))
    streams["topk"].extend(("set", "ready", row) for row in range(min(rows, 2)))
    for row in range(rows):
        if row >= 2:
            streams["topk"].append(("wait", "free", row))
        streams["topk"].extend(
            [("wait", "scored", row), ("topk_begin", "", row), ("topk_end", "", row), ("set", "ack", row)]
        )
        if row + 2 < rows:
            streams["topk"].append(("set", "ready", row + 2))
    for row in range(rows):
        if row >= 2:
            streams["cube"].extend([("wait", "ack", row - 2), ("set", "free", row - 2)])
        streams["cube"].extend(
            [("wait", "ready", row), ("cube_begin", "", row), ("cube_end", "", row), ("set", "scored", row)]
        )
    streams["cube"].extend(("wait", "ack", row) for row in range(max(0, rows - 2), rows))
    # AIV -> AIC MODE2 aggregates one contribution from each subcore. AIC ->
    # AIV broadcasts one independently consumable token to each subcore.
    tokens = {(flag, side): 0 for flag in ("ready", "ack", "scored", "free") for side in ("metadata", "topk")}
    pc = dict.fromkeys(streams, 0)
    slots = [dict(owner=-1, metadataed=False, scored=False, reading_k=False, reading_score=False) for _ in range(2)]
    complete = set()
    while any(pc[actor] < len(streams[actor]) for actor in streams):
        runnable = []
        for actor, actions in streams.items():
            if pc[actor] == len(actions):
                continue
            action, flag, _ = actions[pc[actor]]
            if action != "wait":
                runnable.append(actor)
            elif actor == "cube":
                if all(tokens[flag, side] for side in ("metadata", "topk")):
                    runnable.append(actor)
            elif tokens[flag, actor]:
                runnable.append(actor)
        assert runnable, (rows, seed, pc, tokens)
        actor = rng.choice(runnable)
        action, flag, row = streams[actor][pc[actor]]
        pc[actor] += 1
        slot = slots[row % 2]
        if action in ("set", "wait"):
            sides = ("metadata", "topk") if actor == "cube" else (actor,)
            for side in sides:
                tokens[flag, side] += 1 if action == "set" else -1
                assert tokens[flag, side] >= 0
        elif action == "metadata_begin":
            assert not slot["reading_k"] and not slot["reading_score"]
            assert slot["owner"] < 0 or slot["owner"] in complete
            slot.update(owner=row, metadataed=False, scored=False)
        elif action == "metadata_end":
            assert slot["owner"] == row
            slot["metadataed"] = True
        elif action == "cube_begin":
            assert slot["owner"] == row and slot["metadataed"]
            slot["reading_k"] = True
        elif action == "cube_end":
            assert slot["owner"] == row and slot["reading_k"]
            slot.update(reading_k=False, scored=True)
        elif action == "topk_begin":
            assert slot["owner"] == row and slot["scored"]
            slot["reading_score"] = True
        elif action == "topk_end":
            assert slot["owner"] == row and slot["reading_score"]
            slot["reading_score"] = False
            complete.add(row)
    assert complete == set(range(rows))
    assert not any(tokens.values()), tokens


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 4, 5, 16, 33, 129, 513])
def test_two_slot_protocol(rows):
    for seed in range(128):
        simulate(rows, seed)
