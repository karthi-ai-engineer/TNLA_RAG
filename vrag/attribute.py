"""P3.5 — speaker attribution: fuse evidence into member identities.

The rule this module exists to enforce: **a name is never guessed.**  Every
attribution must trace to at least one piece of checkable evidence, and every
name must resolve against the 227-member roster.  "unknown" is a first-class
result that survives to the UI.

Evidence, strongest first:
  chyron        the broadcast printed the name on screen while they spoke (P3)
  introduction  the Speaker announced them by name/role just before (LLM pass
                over the transcript, then roster-resolved in code)
  role_address  the segment text addresses/mentions an office whose sole
                holder is known (e.g. எதிர்க்கட்சித் தலைவர்) — weak, only used
                to corroborate, never to attribute alone

Fusion is deterministic code.  The LLM's only job is reading: "which segments
announce an upcoming speaker, and what name/role do they say?"
"""
from __future__ import annotations

import logging
from collections import Counter

from vrag import config, gemini
from vrag.cache import read_json, write_json
from vrag.roster import ROLE_TAMIL, RosterIndex

log = logging.getLogger(__name__)


class AttributionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Turn building — group segments into continuous same-voice spans
# ---------------------------------------------------------------------------
def build_turns(segments: list[dict]) -> list[dict]:
    """Consecutive segments with the same chunk-local voice label = one turn.

    Voice labels do not cross chunk boundaries (the model cannot know that
    chunk 3's V1 is chunk 2's V2), so turns break at chunk edges too.  Turns
    that in reality continue across a boundary are re-joined later, after
    naming, when both sides resolved to the same member.
    """
    turns: list[dict] = []
    for seg in segments:
        if turns and seg["voice_local"] == turns[-1]["voice_local"] \
                and not seg["new_speaker"]:
            turns[-1]["segment_ids"].append(seg["id"])
            turns[-1]["t_end"] = seg["t_end"]
        else:
            turns.append({
                "turn_index": len(turns),
                "voice_local": seg["voice_local"],
                "t_start": seg["t_start"],
                "t_end": seg["t_end"],
                "segment_ids": [seg["id"]],
            })
    return turns


# ---------------------------------------------------------------------------
# Evidence 1: chyrons (from frames.json)
# ---------------------------------------------------------------------------
def chyron_evidence(turn: dict, frames: list[dict], idx: RosterIndex) -> list[dict]:
    """Roster-resolved chyron sightings that fall inside this turn's window."""
    out = []
    for f in frames:
        if not f["chyron_name"]:
            continue
        if not (turn["t_start"] - 1.0 <= f["t"] <= turn["t_end"] + 1.0):
            continue
        member = idx.find(f["chyron_name"])
        if member is None and f["chyron_role"]:
            member = _member_by_role(f["chyron_role"], idx)
        if member:
            out.append({"kind": "chyron", "member_id": member["member_id"],
                        "t": f["t"], "raw": f["chyron_name"], "frame": f["path"]})
        else:
            log.debug("chyron %r at %.0fs resolved to no member", f["chyron_name"], f["t"])
    return out


# Tamil portfolio terms -> the keyword that appears in the roster's English
# "Minister — <portfolio>" role strings.  Lets a portfolio-specific
# announcement ("மாண்புமிகு பொதுப்பணித்துறை அமைச்சர்") resolve to its single
# holder even though "Minister" alone never can.
PORTFOLIO_TAMIL = {
    "பொதுப்பணி": "Public Works",
    "நிதி": "Finance",
    "பள்ளிக் கல்வி": "School Education",
    "உயர் கல்வி": "Higher Education",
    "வேளாண்": "Agriculture",
    "மின்சார": "Electricity",
    "போக்குவரத்து": "Transport",
    "தொழில்": "Industries",
    "வருவாய்": "Revenue",
    "சட்டத்துறை": "Law",
    "ஊரக வளர்ச்சி": "Rural Development",
    "கூட்டுறவு": "Co-operation",
    "சுகாதார": "Health",
    "மக்கள் நல்வாழ்வு": "Health",
    "உள்ளாட்சி": "Local Administration",
    "நகராட்சி": "Municipal",
}


def _holds_role(member: dict, en_role: str) -> bool:
    """Exact office match: 'Deputy Leader of the Opposition' must NOT count as
    holding 'Leader of the Opposition' (substring matching lost us exactly
    that resolution)."""
    want = en_role.lower()
    for r in member["current_roles"]:
        rl = r.lower().strip()
        if rl == want or rl.startswith((want + " —", want + " -", want + ",")):
            return True
    return False


def _member_by_role(role_text: str, idx: RosterIndex) -> dict | None:
    """Resolve an office name (either script) to its single current holder."""
    role_text_l = role_text.lower()

    # Portfolio-specific minister first — more specific beats more generic.
    for ta_term, en_keyword in PORTFOLIO_TAMIL.items():
        if ta_term in role_text:
            holders = [m for m in idx.roster["members"]
                       if any("minister" in r.lower() and en_keyword.lower() in r.lower()
                              for r in m["current_roles"])]
            if len(holders) == 1:
                return holders[0]

    for en_role, ta_forms in ROLE_TAMIL.items():
        if en_role.lower() in role_text_l or any(t in role_text for t in ta_forms):
            if en_role == "Minister":     # 35 holders — never unique, never resolves
                continue
            holders = [m for m in idx.roster["members"] if _holds_role(m, en_role)]
            if len(holders) == 1:
                return holders[0]
    return None


# ---------------------------------------------------------------------------
# Evidence 2: introductions (one LLM reading pass over the transcript)
# ---------------------------------------------------------------------------
_INTRO_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "introductions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "segment_id": {"type": "STRING"},
                    "announced_name": {"type": "STRING"},
                    "announced_role": {"type": "STRING"},
                    "quote": {"type": "STRING"},
                },
                "required": ["segment_id", "announced_name", "announced_role", "quote"],
            },
        }
    },
    "required": ["introductions"],
}

_INTRO_PROMPT = """This is a Tamil transcript of a Tamil Nadu Legislative Assembly \
session, as numbered segments. In assembly procedure, the Speaker (சபாநாயகர்) \
calls each member before they speak, and members are announced by name and/or \
office.

Find every segment that ANNOUNCES OR INVITES a person to speak next (e.g. \
"... அவர்கள் பேசலாம்", "மாண்புமிகு ... அவர்கள்", calling on a member or minister). \
For each, return:
- segment_id: the segment containing the announcement
- announced_name: the person's NAME exactly as spoken (Tamil script), empty if
  only an office is mentioned
- announced_role: the office/title mentioned (e.g. அமைச்சர், எதிர்க்கட்சித் தலைவர்),
  empty if none
- quote: the exact words of the announcement

Only report announcements actually present in the text. Mentioning a person in
debate is NOT an announcement — only calls/invitations to speak count.

Transcript:
{transcript}"""


INTRO_BATCH = 150   # segments per call — a 3.5h session in ONE call blew the
                    # output cap (thinking tokens count against it); windows
                    # keep every call small no matter how long the video is.


def find_introductions(client, segments: list[dict]) -> list[dict]:
    from vrag.parallel import thread_map

    batches = [segments[i:i + INTRO_BATCH]
               for i in range(0, len(segments), INTRO_BATCH)]
    log.info("attribute: scanning for announcements in %d batch(es)", len(batches))

    def run_batch(batch: list[dict]) -> list[dict]:
        lines = "\n".join(f"[{s['id']}] {s['text_ta']}" for s in batch)
        out = client.generate_json(
            [gemini.text_part(_INTRO_PROMPT.format(transcript=lines))],
            schema=_INTRO_SCHEMA,
            timeout=config.TIMEOUT_TEXT,
            namespace="introductions",
            max_output_tokens=32768,
        )
        return out.get("introductions") or []

    results = thread_map(run_batch, batches, workers=client.max_workers,
                         desc="introductions")
    known_ids = {s["id"] for s in segments}
    seen: set[str] = set()
    intros = []
    for batch_out in results:
        for i in batch_out:
            if i["segment_id"] in known_ids and i["segment_id"] not in seen:
                seen.add(i["segment_id"])
                intros.append(i)
    log.info("attribute: %d introduction announcements found", len(intros))
    return intros


def introduction_evidence(turn_i: int, turns: list[dict], intros: list[dict],
                          idx: RosterIndex) -> list[dict]:
    """An announcement in the PREVIOUS turn names THIS turn's speaker."""
    if turn_i == 0:
        return []
    prev_ids = set(turns[turn_i - 1]["segment_ids"])
    out = []
    for intro in intros:
        if intro["segment_id"] not in prev_ids:
            continue
        member = idx.find(intro["announced_name"]) if intro["announced_name"] else None
        if member is None and intro["announced_role"]:
            member = _member_by_role(intro["announced_role"], idx)
        if member:
            out.append({"kind": "introduction", "member_id": member["member_id"],
                        "segment_id": intro["segment_id"], "raw": intro["quote"]})
    return out


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def attribute_turns(turns: list[dict], frames: list[dict], intros: list[dict],
                    idx: RosterIndex) -> None:
    """Attach speaker attribution to every turn, in place.

    Policy: chyron outranks introduction; agreement raises confidence;
    disagreement keeps the chyron but flags the conflict; no evidence means
    UNKNOWN — never a guess.
    """
    for i, turn in enumerate(turns):
        ev = chyron_evidence(turn, frames, idx) \
             + introduction_evidence(i, turns, intros, idx)
        if not ev:
            turn["speaker"] = {"member_id": None, "name": None,
                               "evidence": [], "confidence": 0.0}
            continue

        votes = Counter(e["member_id"] for e in ev)
        (winner, n_win), = votes.most_common(1)
        kinds = sorted({e["kind"] for e in ev if e["member_id"] == winner})
        conflict = len(votes) > 1

        if conflict:
            chyron_ids = {e["member_id"] for e in ev if e["kind"] == "chyron"}
            if len(chyron_ids) == 1:            # trust the printed name
                winner = next(iter(chyron_ids))
                kinds = sorted({e["kind"] for e in ev if e["member_id"] == winner})
            log.warning("turn %d (%.0f-%.0fs): conflicting evidence %s -> kept %s",
                        i, turn["t_start"], turn["t_end"], dict(votes), winner)

        member = idx.by_id[winner]
        confidence = 0.95 if len(kinds) > 1 else (0.85 if "chyron" in kinds else 0.75)
        if conflict:
            confidence = min(confidence, 0.6)
        turn["speaker"] = {
            "member_id": member["member_id"],
            "name": member["name"],
            "tamil_name": member.get("tamil_name"),
            "party_code": member["party_code"],
            "roles": member["current_roles"],
            "evidence": [{k: v for k, v in e.items() if k != "kind"} | {"kind": e["kind"]}
                         for e in ev if e["member_id"] == winner],
            "evidence_kinds": kinds,
            "conflict": conflict,
            "confidence": confidence,
        }


def apply_procedure_rules(turns: list[dict], intros: list[dict],
                          idx: RosterIndex) -> None:
    """Two attribution rules that come from assembly procedure itself.

    1. The person who CALLS members to speak is the presiding Speaker — so an
       otherwise-unnamed turn containing an announcement is the Speaker's.
    2. A turn split only by OUR chunk boundary (different chunk prefix,
       adjacent in time) is the same person still talking; it inherits the
       floor-holder at reduced confidence.  A voice change detected WITHIN a
       chunk is a real change and never inherits.
    """
    intro_seg_ids = {i["segment_id"] for i in intros}
    presiding = _member_by_role("சபாநாயகர்", idx)   # unique holder of Speaker

    def has_announcement(turn: dict) -> bool:
        return any(sid in intro_seg_ids for sid in turn["segment_ids"])

    if presiding:
        for turn in turns:
            if not turn["speaker"]["member_id"] and has_announcement(turn):
                turn["speaker"] = {
                    "member_id": presiding["member_id"],
                    "name": presiding["name"],
                    "tamil_name": presiding.get("tamil_name"),
                    "party_code": presiding["party_code"],
                    "roles": presiding["current_roles"],
                    "evidence": [{"kind": "procedure",
                                  "raw": "turn contains the call to speak"}],
                    "evidence_kinds": ["procedure"],
                    "conflict": False,
                    "confidence": 0.7,
                }
    else:
        log.warning("attribute: no unique Speaker in roster — announcement "
                    "turns stay unknown")

    def chunk_of(turn: dict) -> str:
        return turn["voice_local"].split(":")[0]

    for i in range(1, len(turns)):
        cur, prev = turns[i], turns[i - 1]
        if cur["speaker"]["member_id"] or has_announcement(cur):
            continue
        if chunk_of(cur) == chunk_of(prev):
            continue                        # real voice change, not our split
        if cur["t_start"] - prev["t_end"] > 10.0:
            continue                        # too far apart to assume continuity
        src = prev["speaker"]
        if not src["member_id"] or "procedure" in src.get("evidence_kinds", []):
            continue                        # never continue "the Speaker's" turn
        cur["speaker"] = {
            **{k: src[k] for k in ("member_id", "name", "tamil_name",
                                   "party_code", "roles")},
            "evidence": [{"kind": "continuation",
                          "raw": f"chunk-boundary split from turn {prev['turn_index']}"}],
            "evidence_kinds": ["continuation"],
            "conflict": False,
            "confidence": round(max(0.5, src["confidence"] - 0.1), 2),
        }


def merge_adjacent_same_speaker(turns: list[dict]) -> list[dict]:
    """Re-join turns split by chunk boundaries once both sides have a name.

    Only merges when both sides are attributed to the SAME member and are
    within 30s of each other — an unknown never absorbs into a named turn.
    """
    merged: list[dict] = []
    for t in turns:
        prev = merged[-1] if merged else None
        if (prev and prev["speaker"]["member_id"]
                and prev["speaker"]["member_id"] == t["speaker"]["member_id"]
                and t["t_start"] - prev["t_end"] <= 30.0):
            prev["segment_ids"].extend(t["segment_ids"])
            prev["t_end"] = t["t_end"]
            prev["speaker"]["evidence"].extend(t["speaker"]["evidence"])
            prev["speaker"]["confidence"] = max(prev["speaker"]["confidence"],
                                                t["speaker"]["confidence"])
        else:
            merged.append(t)
    for i, t in enumerate(merged):
        t["turn_index"] = i
    return merged


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def attribute_video(client, video_id: str) -> dict:
    art_dir = config.artifact_dir(video_id)
    seg_art = read_json(art_dir / "segments.json")
    frames_art = read_json(art_dir / "frames.json")
    segments = seg_art["segments"]
    frames = frames_art["frames"]
    idx = RosterIndex()

    turns = build_turns(segments)
    log.info("attribute: %d voice turns from %d segments", len(turns), len(segments))

    intros = find_introductions(client, segments)
    attribute_turns(turns, frames, intros, idx)
    apply_procedure_rules(turns, intros, idx)
    turns = merge_adjacent_same_speaker(turns)

    # Write speaker back onto each segment — the unit everything cites.
    by_seg: dict[str, dict] = {}
    for t in turns:
        for sid in t["segment_ids"]:
            by_seg[sid] = t
    for s in segments:
        t = by_seg.get(s["id"])
        s["speaker"] = t["speaker"] if t else {
            "member_id": None, "name": None, "evidence": [], "confidence": 0.0}
        s["turn_index"] = t["turn_index"] if t else None

    named = [t for t in turns if t["speaker"]["member_id"]]
    cast_ids = sorted({t["speaker"]["member_id"] for t in named})
    cast = [idx.by_id[m] for m in cast_ids]
    named_secs = sum(t["t_end"] - t["t_start"] for t in named)
    total_secs = sum(t["t_end"] - t["t_start"] for t in turns)
    log.info("attribute: %d/%d turns named (%.0f%% of speech time), cast of %d",
             len(named), len(turns),
             100 * named_secs / total_secs if total_secs else 0, len(cast))

    result = {
        "video_id": video_id,
        "turn_count": len(turns),
        "named_turn_count": len(named),
        "named_speech_ratio": round(named_secs / total_secs, 3) if total_secs else 0.0,
        "cast": [{"member_id": m["member_id"], "name": m["name"],
                  "tamil_name": m.get("tamil_name"), "party_code": m["party_code"],
                  "roles": m["current_roles"], "portrait": m.get("portrait")}
                 for m in cast],
        "turns": turns,
    }
    write_json(art_dir / "turns.json", result)
    write_json(art_dir / "segments.json", seg_art)   # now speaker-enriched
    log.info("attribute: wrote turns.json and updated segments.json")
    return result
