"""P1.5 — the member roster: 234 MLAs with Tamil aliases and portraits.

Why this exists
---------------
Speaker attribution never guesses from an open vocabulary.  Every name the
pipeline assigns must resolve against this closed list of sitting members —
a wrong-but-plausible name is the one error class this demo cannot afford.

The source data (`members.json`, built from elections.tn.gov.in +
tnla.neva.gov.in) is entirely romanized: "UDHAYANIDHI STALIN".  The transcript
is Tamil: "உதயநிதி ஸ்டாலின்".  The one thing we must generate is the bridge —
Tamil script forms for every member, produced once by Gemini and cached.

Per-video, only ~20-40 of the 234 actually appear.  Downstream phases first
narrow to that "active cast" (names found in chyrons + transcript), then
attribute against the cast only.  This module provides the lookup machinery
for both steps.
"""
from __future__ import annotations

import difflib
import logging
import pathlib
import re
import shutil
from typing import Any, Iterable

from vrag import config
from vrag.cache import read_json, write_json

log = logging.getLogger(__name__)

# Committed with the repo (curated data, not a generated artifact): a fresh
# clone has the full roster + portraits and needs no external source or API
# call to use them.  scripts/p15_roster.py rebuilds it from source if the
# assembly composition ever changes.
ROSTER_DIR = config.ROOT / "assets" / "roster"
ROSTER_JSON = ROSTER_DIR / "roster.json"
PORTRAIT_DIR = ROSTER_DIR / "images"

# Offices as they are actually said on the floor / shown on chyrons.
# Static because they are stable Tamil political vocabulary, not per-member data.
ROLE_TAMIL = {
    "Chief Minister": ["முதலமைச்சர்", "முதல்வர்"],
    "Deputy Chief Minister": ["துணை முதலமைச்சர்", "துணை முதல்வர்"],
    "Speaker": ["சபாநாயகர்", "அவைத்தலைவர்", "பேரவைத் தலைவர்"],
    "Deputy Speaker": ["துணை சபாநாயகர்"],
    "Leader of the Opposition": ["எதிர்க்கட்சித் தலைவர்"],
    "Minister": ["அமைச்சர்"],
}


class RosterError(RuntimeError):
    """Roster build/lookup failure.  Loud by design."""


# ---------------------------------------------------------------------------
# Build (runs once; re-run only if the source data changes)
# ---------------------------------------------------------------------------
def build_roster(source_members_json: pathlib.Path,
                 source_images_dir: pathlib.Path,
                 copy_portraits: bool = True) -> dict:
    """Convert the external members.json into our project roster (sans Tamil).

    Tamil aliases are added by `add_tamil_aliases` in a separate, cached step,
    so a source-data refresh doesn't force re-paying for transliteration.
    """
    src = read_json(source_members_json)
    members_in = src.get("members") or []
    if len(members_in) != 234:
        raise RosterError(f"Expected 234 members, got {len(members_in)} — wrong file?")

    ROSTER_DIR.mkdir(parents=True, exist_ok=True)
    if copy_portraits:
        PORTRAIT_DIR.mkdir(parents=True, exist_ok=True)

    members_out: list[dict] = []
    vacant = 0
    for m in members_in:
        if m.get("membership_state") == "VACANT" or m.get("name") == "VACANT":
            # A vacant seat has no sitting member; its portrait shows the
            # FORMER member, which must never be matchable to current speech.
            vacant += 1
            continue

        portrait_rel = None
        local = (m.get("image") or {}).get("local_path")
        if local:
            src_img = source_members_json.parent.parent.parent / local
            if src_img.exists():
                portrait_rel = f"images/{src_img.name}"
                dst = PORTRAIT_DIR / src_img.name
                if copy_portraits and not dst.exists():
                    shutil.copy2(src_img, dst)
            else:
                log.warning("roster: portrait missing for %s: %s", m["name"], src_img)

        members_out.append({
            "member_id": m["member_id"],
            "name": m["name"].strip(),
            "constituency_number": m["constituency_number"],
            "constituency_name": m["constituency_name"],
            "party": m["party"],
            "party_code": (m.get("metadata") or {}).get("partycode", ""),
            "membership_state": m["membership_state"],   # CURRENT (govt) / OPPONENT
            "current_roles": m.get("current_roles") or [],
            "former_roles": m.get("former_roles") or [],
            "portrait": portrait_rel,
            # Filled in by add_tamil_aliases:
            "tamil_name": None,
            "tamil_aliases": [],
            "en_aliases": [a for a in (m.get("aliases") or []) if a],
        })

    roster = {
        "schema_version": 1,
        "assembly": src.get("assembly"),
        "source_generated_at": src.get("generated_at"),
        "sitting_members": len(members_out),
        "vacant_seats": vacant,
        "members": members_out,
    }
    write_json(ROSTER_JSON, roster)
    log.info("roster: built %d sitting members (%d vacant) -> %s",
             len(members_out), vacant, ROSTER_JSON)
    return roster


# ---------------------------------------------------------------------------
# Tamil alias generation (one-time Gemini pass, cached like every call)
# ---------------------------------------------------------------------------
_ALIAS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "members": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "member_id": {"type": "STRING"},
                    "tamil_name": {"type": "STRING"},
                    "tamil_aliases": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "en_aliases": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["member_id", "tamil_name", "tamil_aliases", "en_aliases"],
            },
        }
    },
    "required": ["members"],
}

_ALIAS_PROMPT = """You are transliterating the names of sitting Tamil Nadu Legislative \
Assembly members into Tamil script, for matching against a Tamil transcript of \
assembly proceedings.

For each member below, return:
- tamil_name: the full name in Tamil script, as a Tamil news channel would write it. \
For well-known politicians use the established Tamil spelling of their name; for \
others produce a faithful standard transliteration. Keep initials as Latin letters \
followed by a dot (e.g. "மு.க." style) only when that is the conventional form; \
otherwise transliterate them.
- tamil_aliases: 1-3 other Tamil forms they are commonly referred to by IN THE \
ASSEMBLY OR NEWS: surname-only, popular short name, name without initials. Do NOT \
invent honorifics or office titles — offices are handled separately.
- en_aliases: 0-2 common English/romanized variants (different initial order, \
common press spelling). Empty list if none.

Return every member_id you were given, exactly once.

Members (member_id | romanized name | party | constituency):
{rows}"""

_TAMIL_RE = re.compile("[஀-௿]")


def add_tamil_aliases(client: Any, roster: dict | None = None,
                      batch_size: int = 30) -> dict:
    """Fill tamil_name / tamil_aliases for every member, batched and validated."""
    from vrag.parallel import thread_map

    roster = roster or load_roster()
    members = roster["members"]
    todo = [m for m in members if not m.get("tamil_name")]
    if not todo:
        log.info("roster: all %d members already have Tamil names", len(members))
        return roster

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    log.info("roster: generating Tamil aliases for %d members in %d batches",
             len(todo), len(batches))

    def run_batch(batch: list[dict]) -> dict[str, dict]:
        rows = "\n".join(
            f"{m['member_id']} | {m['name']} | {m['party_code']} | {m['constituency_name']}"
            for m in batch)
        from vrag import gemini
        out = client.generate_json(
            [gemini.text_part(_ALIAS_PROMPT.format(rows=rows))],
            schema=_ALIAS_SCHEMA,
            timeout=config.TIMEOUT_TEXT,
            namespace="roster_tamil",
            max_output_tokens=16384,
        )
        got = {e["member_id"]: e for e in out.get("members", [])}
        # Validate hard: every id answered, every name actually in Tamil script.
        for m in batch:
            e = got.get(m["member_id"])
            if e is None:
                raise RosterError(f"Gemini omitted {m['member_id']} ({m['name']})")
            if not _TAMIL_RE.search(e.get("tamil_name") or ""):
                raise RosterError(
                    f"No Tamil glyphs in tamil_name for {m['name']}: {e.get('tamil_name')!r}")
        return got

    results = thread_map(run_batch, batches, workers=client.max_workers,
                         desc="roster-tamil")
    merged: dict[str, dict] = {}
    for r in results:
        merged.update(r)

    for m in members:
        e = merged.get(m["member_id"])
        if e:
            m["tamil_name"] = e["tamil_name"].strip()
            m["tamil_aliases"] = [a.strip() for a in e["tamil_aliases"] if a.strip()]
            m["en_aliases"] = sorted(
                {*m.get("en_aliases", []), *(a.strip() for a in e["en_aliases"] if a.strip())})

    write_json(ROSTER_JSON, roster)
    log.info("roster: Tamil aliases written for %d members", len(merged))
    return roster


# ---------------------------------------------------------------------------
# Lookup — used by P2 (prompt seeding), P3 (chyron matching), P3.5 (attribution)
# ---------------------------------------------------------------------------
def load_roster() -> dict:
    if not ROSTER_JSON.exists():
        raise RosterError(
            f"No roster at {ROSTER_JSON}. Run:  python scripts/p15_roster.py")
    return read_json(ROSTER_JSON)


def _norm(s: str) -> str:
    """Normalise a name for matching: case, dots, extra spaces, honorifics."""
    s = s.upper()
    s = re.sub(r"\b(DR|THIRU|TMT|SELVI|MR|MRS|MS)\b\.?", " ", s)
    s = re.sub(r"[.​‌‍]", " ", s)
    return " ".join(s.split())


class RosterIndex:
    """Alias -> member index over both scripts, with fuzzy fallback.

    Matching policy: exact normalised hit first; else a high-cutoff fuzzy
    match; else None.  Returning None is correct behaviour, not failure —
    'unknown speaker' must survive all the way to the UI.
    """

    def __init__(self, roster: dict | None = None):
        self.roster = roster or load_roster()
        self.by_id: dict[str, dict] = {m["member_id"]: m for m in self.roster["members"]}
        self._exact: dict[str, str] = {}       # normalised alias -> member_id
        for m in self.roster["members"]:
            aliases = [m["name"], *m.get("en_aliases", [])]
            if m.get("tamil_name"):
                aliases.append(m["tamil_name"])
            aliases.extend(m.get("tamil_aliases", []))
            for a in aliases:
                key = _norm(a)
                if not key:
                    continue
                if key in self._exact and self._exact[key] != m["member_id"]:
                    # Two members sharing an alias: drop it — ambiguous aliases
                    # must not silently pick one of them.
                    log.debug("roster: ambiguous alias %r dropped", a)
                    self._exact[key] = ""
                    continue
                self._exact.setdefault(key, m["member_id"])
        self._keys = [k for k, v in self._exact.items() if v]

    def find(self, name: str, fuzzy_cutoff: float = 0.88) -> dict | None:
        key = _norm(name)
        if not key:
            return None
        mid = self._exact.get(key)
        if mid:
            return self.by_id[mid]
        close = difflib.get_close_matches(key, self._keys, n=1, cutoff=fuzzy_cutoff)
        if close:
            return self.by_id[self._exact[close[0]]]
        return None

    def find_all(self, names: Iterable[str]) -> dict[str, dict | None]:
        return {n: self.find(n) for n in names}
