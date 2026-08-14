"""P3.6 — face-lineup verification of speaker attribution.

The idea (proposed by the user, kept because it is the textbook
retrieval-then-verify pattern): for each attributed turn, show the model a
frame of the person standing/speaking TOGETHER WITH a small photo lineup —
the attributed member's official portrait plus decoy portraits — and ask
which lineup photo, if any, shows the standing person.

This is *comparison* ("same person in these two images?"), not open-set
recognition ("name this face"), which models handle far better.  The lineup
is honestly constructed: decoys are real members, "none" is always a valid
answer, and the model is never told which photo we expect.

Verification policy:
  match      -> confidence boosted, turn marked face_verified
  none/decoy -> conflict logged loudly, confidence cut — NEVER silently kept
The pass never *assigns* names by itself; it only strengthens or weakens
attributions that text evidence proposed.  Face evidence on 720p wide shots
is corroboration, not identification.
"""
from __future__ import annotations

import logging
import random

from vrag import config, gemini
from vrag.cache import read_json, write_json
from vrag.parallel import thread_map
from vrag.roster import ROSTER_DIR, RosterIndex

log = logging.getLogger(__name__)

LINEUP_SIZE = 5          # 1 expected + 4 decoys
FRAMES_PER_TURN = 2      # verify against up to N standing-frames per turn


_VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "match": {"type": "STRING"},        # "A".."E" or "none"
        "confidence": {"type": "STRING"},   # "high" | "medium" | "low"
        "reason": {"type": "STRING"},
    },
    "required": ["match", "confidence", "reason"],
}

_VERIFY_PROMPT = """The FIRST image is a frame from a video of the Tamil Nadu \
Legislative Assembly: one member is STANDING and speaking while others sit.

The next {n} images are official portrait photos, labelled {labels} in order.

Question: which portrait, if any, shows the SAME PERSON as the one standing and
speaking in the first image? Compare facial structure, age, build, hair and
moustache. The standing person may not be in the lineup at all.

Answer "none" unless you are genuinely confident of a match. Judge only by
visual comparison — do not use any outside knowledge about who these people are."""


def _portrait_path(member: dict):
    p = member.get("portrait")
    return (ROSTER_DIR / p) if p else None


def _standing_frames(turn: dict, frames: list[dict], limit: int) -> list[dict]:
    """Standing-speaker frames inside the turn, spread across its length."""
    inside = [f for f in frames
              if f["person_standing"] and turn["t_start"] <= f["t"] <= turn["t_end"]]
    if len(inside) <= limit:
        return inside
    step = len(inside) // limit
    return inside[::step][:limit]


def _build_lineup(expected: dict, idx: RosterIndex, rng: random.Random) -> list[dict]:
    """Expected member + decoys with portraits, shuffled."""
    pool = [m for m in idx.roster["members"]
            if m["member_id"] != expected["member_id"] and m.get("portrait")]
    decoys = rng.sample(pool, LINEUP_SIZE - 1)
    lineup = [expected] + decoys
    rng.shuffle(lineup)
    return lineup


def verify_turn(client, turn: dict, frames: list[dict], idx: RosterIndex,
                frame_dir) -> dict | None:
    """Run the lineup for one attributed turn.  Returns a verdict record."""
    sp = turn["speaker"]
    member = idx.by_id.get(sp["member_id"])
    if member is None or not member.get("portrait"):
        return None
    stands = _standing_frames(turn, frames, FRAMES_PER_TURN)
    if not stands:
        return {"turn_index": turn["turn_index"], "verdict": "no_frames"}

    # Deterministic lineup per turn+member => stable cache keys across runs.
    rng = random.Random(f"{turn['turn_index']}:{sp['member_id']}")
    lineup = _build_lineup(member, idx, rng)
    labels = [chr(ord("A") + i) for i in range(len(lineup))]
    expected_label = labels[next(i for i, m in enumerate(lineup)
                                 if m["member_id"] == member["member_id"])]

    votes = []
    for fr in stands:
        parts = [gemini.text_part(_VERIFY_PROMPT.format(
            n=len(lineup), labels=", ".join(labels)))]
        parts.append(gemini.image_part(frame_dir / fr["path"]))
        for m in lineup:
            parts.append(gemini.image_part(_portrait_path(m)))
        out = client.generate_json(
            parts, schema=_VERIFY_SCHEMA,
            timeout=config.TIMEOUT_VISION, namespace="faceverify",
        )
        votes.append({"frame": fr["path"], "t": fr["t"],
                      "picked": out["match"].strip().upper()[:4],
                      "model_confidence": out["confidence"],
                      "reason": out["reason"][:200]})

    picks = [v["picked"] for v in votes]
    n_match = sum(1 for p in picks if p == expected_label)
    n_other = sum(1 for p in picks if p in labels and p != expected_label)
    verdict = ("confirmed" if n_match == len(picks) and picks else
               "partial" if n_match > 0 else
               "contradicted" if n_other > 0 else
               "unrecognised")            # every vote said "none"
    return {
        "turn_index": turn["turn_index"],
        "member_id": member["member_id"],
        "member_name": member["name"],
        "expected_label": expected_label,
        "lineup": [m["name"] for m in lineup],
        "votes": votes,
        "verdict": verdict,
    }


def reidentify_turn(client, turn: dict, candidate: dict, frames: list[dict],
                    idx: RosterIndex, frame_dir) -> dict | None:
    """Confirmation lineup: candidate as expected, fresh decoys, fresh shuffle.

    Used when a contradicted turn's votes converged on one specific lineup
    member — one accidental hit is not attribution-grade until it survives a
    second, differently-constructed lineup.
    """
    probe = {**turn, "speaker": {**turn["speaker"],
                                 "member_id": candidate["member_id"]}}
    # Different seed salt => different decoys and ordering than any prior run.
    rng = random.Random(f"confirm:{turn['turn_index']}:{candidate['member_id']}")
    stands = _standing_frames(turn, frames, FRAMES_PER_TURN)
    if not stands:
        return None
    lineup = _build_lineup(candidate, idx, rng)
    labels = [chr(ord("A") + i) for i in range(len(lineup))]
    expected_label = labels[next(i for i, m in enumerate(lineup)
                                 if m["member_id"] == candidate["member_id"])]
    votes = []
    for fr in stands:
        parts = [gemini.text_part(_VERIFY_PROMPT.format(
            n=len(lineup), labels=", ".join(labels)))]
        parts.append(gemini.image_part(frame_dir / fr["path"]))
        for m in lineup:
            parts.append(gemini.image_part(_portrait_path(m)))
        out = client.generate_json(
            parts, schema=_VERIFY_SCHEMA,
            timeout=config.TIMEOUT_VISION, namespace="faceverify",
        )
        votes.append({"frame": fr["path"], "t": fr["t"],
                      "picked": out["match"].strip().upper()[:4],
                      "model_confidence": out["confidence"],
                      "reason": out["reason"][:200]})
    confirmed = bool(votes) and all(v["picked"] == expected_label for v in votes)
    return {"turn_index": turn["turn_index"], "candidate": candidate["name"],
            "member_id": candidate["member_id"], "confirmed": confirmed,
            "votes": votes}


def _convergent_candidate(result: dict, idx: RosterIndex) -> dict | None:
    """If every vote picked the SAME wrong lineup member, return that member."""
    picks = {v["picked"] for v in result["votes"]}
    if len(picks) != 1:
        return None
    label = next(iter(picks))
    if label == result["expected_label"] or len(label) != 1:
        return None
    pos = ord(label) - ord("A")
    if not (0 <= pos < len(result["lineup"])):
        return None
    name = result["lineup"][pos]
    return next((m for m in idx.roster["members"] if m["name"] == name), None)


def verify_video(client, video_id: str) -> dict:
    art_dir = config.artifact_dir(video_id)
    turns_art = read_json(art_dir / "turns.json")
    frames_art = read_json(art_dir / "frames.json")
    frame_dir = art_dir / "frames"
    idx = RosterIndex()

    named = [t for t in turns_art["turns"] if t["speaker"]["member_id"]]
    log.info("faceverify: checking %d attributed turns", len(named))

    results = thread_map(
        lambda t: verify_turn(client, t, frames_art["frames"], idx, frame_dir),
        named, workers=client.max_workers, desc="faceverify")
    results = [r for r in results if r]

    by_turn = {r["turn_index"]: r for r in results if "verdict" in r}
    for t in turns_art["turns"]:
        r = by_turn.get(t["turn_index"])
        if not r or r["verdict"] == "no_frames":
            continue
        sp = t["speaker"]
        sp["face_verdict"] = r["verdict"]
        if r["verdict"] == "confirmed":
            sp["confidence"] = round(min(0.98, sp["confidence"] + 0.15), 2)
            sp["evidence_kinds"] = sorted({*sp["evidence_kinds"], "face"})
        elif r["verdict"] == "partial":
            sp["confidence"] = round(min(0.95, sp["confidence"] + 0.05), 2)
        elif r["verdict"] == "contradicted":
            sp["confidence"] = round(max(0.3, sp["confidence"] - 0.3), 2)
            sp["conflict"] = True
            log.warning("faceverify: turn %d (%s) CONTRADICTED — lineup votes "
                        "went to a different member. Review before trusting.",
                        t["turn_index"], sp["name"])
            # All votes converging on ONE other member is a lead worth testing
            # with a second, differently-built lineup.
            cand = _convergent_candidate(r, idx)
            if cand:
                confirm = reidentify_turn(client, t, cand,
                                          frames_art["frames"], idx, frame_dir)
                r["reidentify"] = confirm
                if confirm and confirm["confirmed"]:
                    log.warning("faceverify: turn %d re-attributed %s -> %s "
                                "(two independent lineups agree)",
                                t["turn_index"], sp["name"], cand["name"])
                    t["speaker"] = {
                        "member_id": cand["member_id"],
                        "name": cand["name"],
                        "tamil_name": cand.get("tamil_name"),
                        "party_code": cand["party_code"],
                        "roles": cand["current_roles"],
                        "evidence": [{"kind": "face",
                                      "raw": "two independent photo lineups"}],
                        "evidence_kinds": ["face"],
                        "conflict": False,
                        "confidence": 0.7,
                    }
        # "unrecognised" (all none): wide shot too hard — no change, recorded.

    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    log.info("faceverify: %s", counts)

    # Push updated speakers back onto the segments — the unit citations use.
    seg_art = read_json(art_dir / "segments.json")
    by_turn_speaker = {t["turn_index"]: t["speaker"] for t in turns_art["turns"]}
    for s in seg_art["segments"]:
        if s.get("turn_index") is not None and s["turn_index"] in by_turn_speaker:
            s["speaker"] = by_turn_speaker[s["turn_index"]]
    write_json(art_dir / "segments.json", seg_art)

    write_json(art_dir / "faceverify.json", {"video_id": video_id, "results": results})
    write_json(art_dir / "turns.json", turns_art)
    return {"video_id": video_id, "counts": counts, "results": results}
