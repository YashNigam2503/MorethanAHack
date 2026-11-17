"""Stage 3: possession evidence arrays. Evidence only -- no labels, no predictions.

Turns the Stage 2 geometry cache (local_cache/tracking_features.npz, written by
stage2_features.py) and the automatic Smart Tagging into explicit per-frame evidence arrays.

This module does NOT produce an H/A/D/X submission, run a possession state machine, smooth over
time, interpolate the ball, apply a control-distance threshold or claim any accuracy. A nearest
team is an observation, not a possession label or a calibrated confidence. A tag is automatic
output with false positives and negatives, and its intervals are phases, not possessions.

Run (human, in a normal terminal, after stage2_features.py):

    python possession_evidence.py

Reads local_cache/tracking_features.npz, the tracking metadata (for team names, via the Stage 2
helper) and the Smart Tagging (via the public loader). Writes local_cache/possession_evidence.npz.
The output is a restricted-data derivative: keep it private and Git-ignored.

Team columns are always 0 = home (first <Team> in the tracking metadata), 1 = away, the same
convention as stage2_features.py and starter/baseline.py.

Output arrays (N = number of video frames)
------------------------------------------
See OUTPUT_SCHEMA below; it is also saved inside the .npz as `schema_json`.
"""
import argparse
import json
import numbers
import sys
from pathlib import Path

import numpy as np

from games import GAME, GAMES, require, source_frames
from stage2_features import read_metadata
from starter.load_smart_tagging import load as load_tags

SCHEMA_VERSION = 1
HOME, AWAY, NO_TEAM = 0, 1, -1

# ---------------------------------------------------------------------------------------------
# Event-code allowlist
# ---------------------------------------------------------------------------------------------
# These are exactly the 14 codes listed in DATA.md ("Smart Tagging" table). DATA.md names them but
# does not define them, so each category below is an interpretation of the code name, kept
# deliberately narrow. Matching is exact after strip() + casefold(); there is no substring rule.
#
# Team-tag meaning per category (the column that receives evidence):
#   attacking_phase       the tagged team is attacking         -> tagged team
#   defensive_phase       the tagged team is defending          -> the OTHER mapped team
#   attacking_transition  the tagged team is acquiring the ball -> tagged team (context only)
#   defensive_transition  the tagged team is losing the ball    -> tagged team (context only)
#   set_piece             a restart; team may be N/A            -> set-piece context, team kept
#                                                                  separately and uninterpreted
#   context_only          documented but given no possession meaning here
ATTACKING_PHASE = "attacking_phase"
DEFENSIVE_PHASE = "defensive_phase"
ATTACKING_TRANSITION = "attacking_transition"
DEFENSIVE_TRANSITION = "defensive_transition"
SET_PIECE = "set_piece"
CONTEXT_ONLY = "context_only"

EVENT_CODES = {
    "build up": ATTACKING_PHASE,
    "progression": ATTACKING_PHASE,
    "ball in final third": ATTACKING_PHASE,
    "creating chances": ATTACKING_PHASE,
    "ball in the box": ATTACKING_PHASE,
    "players in the box": ATTACKING_PHASE,
    "defending in defensive third": DEFENSIVE_PHASE,
    "defending in middle third": DEFENSIVE_PHASE,
    "defending in attacking third": DEFENSIVE_PHASE,
    "attacking transition": ATTACKING_TRANSITION,
    "defensive transition": DEFENSIVE_TRANSITION,
    "set pieces": SET_PIECE,
    # A long ball is a contested aerial action whose outcome the tag does not state, and a goal
    # tag marks an outcome rather than a phase. Both are recorded as context, not team support.
    "long ball": CONTEXT_ONLY,
    "goals": CONTEXT_ONLY,
}

NA_TEAM = "n/a"

OUTPUT_SCHEMA = {
    # geometry (from the Stage 2 cache)
    "frame": ("(N,)", "int32", "none", "Zero-based video frame, exactly 0..N-1."),
    "ball_visible": ("(N,)", "bool", "none", "Ball position observed on this frame."),
    "geometry_available": ("(N, 2)", "bool", "none",
                           "Ball and at least one slot of that team observed (home, away)."),
    "nearest_dist_m": ("(N, 2)", "float32", "+inf when unavailable",
                       "Ball to nearest visible slot of each team, metres (home, away)."),
    "both_teams_available": ("(N,)", "bool", "none", "geometry_available for both teams."),
    "distance_margin_m": ("(N,)", "float32", "NaN unless both teams available",
                          "Away nearest distance minus home nearest distance, metres. "
                          "Positive means home is nearer."),
    "nearest_team": ("(N,)", "int8", "-1 for exact tie or no geometry",
                     "0 if home strictly nearer, 1 if away strictly nearer. An observation, "
                     "not a possession label. Defined even when only one team is visible."),
    "nearest_distance_m": ("(N,)", "float32", "+inf when no geometry",
                           "Distance from ball to the closest available team, metres."),
    "team_visible_slot_count": ("(N, 2)", "int32", "none",
                                "Visible track slots per team. Slots are not unique players."),
    # tactical tags (Smart Tagging, right-exclusive intervals, boolean union of events)
    "tag_attacking_support": ("(N, 2)", "bool", "False = no evidence",
                              "Covered by an attacking-phase tag for that team."),
    "tag_defensive_support": ("(N, 2)", "bool", "False = no evidence",
                              "Covered by a defensive-phase tag for the OTHER team, i.e. "
                              "support for this column's team being the attacking side."),
    "tag_stable_support": ("(N, 2)", "bool", "False = no evidence",
                           "tag_attacking_support OR tag_defensive_support. A union, not a "
                           "count: correlated tags are not independent confirmations."),
    "tag_contradictory_stable_support": ("(N,)", "bool", "none",
                                         "Stable support for both teams on the same frame."),
    "tag_attacking_transition": ("(N, 2)", "bool", "False = no evidence",
                                 "Attacking transition (acquisition) context for that team. "
                                 "Not part of stable support."),
    "tag_defensive_transition": ("(N, 2)", "bool", "False = no evidence",
                                 "Defensive transition (loss) context for that team. "
                                 "Not part of stable support."),
    "tag_set_piece": ("(N,)", "bool", "none",
                      "Inside a SET PIECES interval, whether or not its team is known. Not a "
                      "dead-ball mask: the interval is a window around the restart."),
    "tag_set_piece_team": ("(N, 2)", "bool", "False = unknown or N/A team",
                           "The Team tag of a covering set piece. Recorded, not interpreted."),
    "tag_context_only": ("(N,)", "bool", "none",
                         "Inside a documented code given no possession meaning (LONG BALL, "
                         "GOALS)."),
    "tag_unhandled_code": ("(N,)", "bool", "none",
                           "Inside an event whose code is not in the documented allowlist."),
    # metadata
    "team_names": ("(2,)", "str", "none", "Tracking-metadata names, home then away."),
    "event_codes": ("(K,)", "str", "none", "Allowlisted event codes (normalised)."),
    "event_categories": ("(K,)", "str", "none", "Category of each allowlisted code."),
    "schema_version": ("()", "int32", "none", "Output schema version."),
    "schema_json": ("()", "str", "none", "This table, as JSON."),
    "summary_json": ("()", "str", "none", "Event/team accounting from tactical_evidence()."),
}

GEOMETRY_KEYS = ("frame", "ball_xy_m", "ball_visible", "nearest_dist_m", "nearest_slot",
                 "team_visible_slot_count", "geometry_available", "slot_team", "pitch_m")


# ---------------------------------------------------------------------------------------------
# 1. Geometry
# ---------------------------------------------------------------------------------------------
def _require_dtype(name, array, kind):
    kinds = {"bool": "b", "int": "iu", "float": "f"}[kind]
    if array.dtype.kind not in kinds:
        raise ValueError("{} must have a {} dtype, got {}.".format(name, kind, array.dtype))


def _require_shape(name, array, shape):
    if array.shape != shape:
        raise ValueError("{} must have shape {}, got {}.".format(name, shape, array.shape))


def validate_geometry(features):
    """Check a Stage 2 feature dict and return it as arrays. Raises ValueError on any defect."""
    missing = [key for key in GEOMETRY_KEYS if key not in features]
    if missing:
        raise ValueError("Stage 2 features are missing keys: {}.".format(missing))
    f = {key: np.asarray(features[key]) for key in GEOMETRY_KEYS}

    frame = f["frame"]
    _require_dtype("frame", frame, "int")
    if frame.ndim != 1 or frame.size == 0:
        raise ValueError("frame must be a non-empty 1-D array.")
    n = frame.shape[0]
    if not np.array_equal(frame, np.arange(n)):
        raise ValueError("frame must be exactly 0..N-1 in order (no gaps or duplicates).")

    slot_team = f["slot_team"]
    _require_dtype("slot_team", slot_team, "int")
    if slot_team.ndim != 1 or slot_team.size == 0:
        raise ValueError("slot_team must be a non-empty 1-D array.")
    if not np.all((slot_team == HOME) | (slot_team == AWAY)):
        raise ValueError("slot_team values must be 0 (home) or 1 (away).")
    n_slots = slot_team.shape[0]

    pitch = f["pitch_m"]
    _require_dtype("pitch_m", pitch, "float")
    _require_shape("pitch_m", pitch, (2,))
    if not (np.all(np.isfinite(pitch)) and np.all(pitch > 0)):
        raise ValueError("pitch_m must be two finite positive metre values.")

    ball, visible = f["ball_xy_m"], f["ball_visible"]
    _require_dtype("ball_xy_m", ball, "float")
    _require_shape("ball_xy_m", ball, (n, 2))
    _require_dtype("ball_visible", visible, "bool")
    _require_shape("ball_visible", visible, (n,))
    if not np.array_equal(visible, np.all(np.isfinite(ball), axis=1)):
        raise ValueError("ball_visible disagrees with finite ball_xy_m.")
    if np.any(np.isinf(ball)):
        raise ValueError("ball_xy_m must use NaN, not infinity, for missing geometry.")

    dist, avail = f["nearest_dist_m"], f["geometry_available"]
    _require_dtype("nearest_dist_m", dist, "float")
    _require_shape("nearest_dist_m", dist, (n, 2))
    _require_dtype("geometry_available", avail, "bool")
    _require_shape("geometry_available", avail, (n, 2))
    if np.any(np.isnan(dist)):
        raise ValueError("nearest_dist_m contains NaN; unavailable distances must be +inf.")
    if not np.array_equal(avail, np.isfinite(dist)):
        raise ValueError("geometry_available disagrees with finite nearest_dist_m.")
    if np.any(dist[~avail] != np.inf):
        raise ValueError("Unavailable nearest_dist_m entries must be +inf, not -inf.")
    if np.any(dist[avail] < 0):
        raise ValueError("nearest_dist_m contains negative distances.")
    if np.any(avail & ~visible[:, None]):
        raise ValueError("geometry_available is set on a frame with no visible ball.")

    slot = f["nearest_slot"]
    _require_dtype("nearest_slot", slot, "int")
    _require_shape("nearest_slot", slot, (n, 2))
    if np.any(slot[~avail] != NO_TEAM):
        raise ValueError("nearest_slot must be -1 wherever geometry is unavailable.")
    chosen = slot[avail]
    if np.any((chosen < 0) | (chosen >= n_slots)):
        raise ValueError("nearest_slot refers to a slot outside slot_team.")
    team_column = np.broadcast_to(np.array([HOME, AWAY]), (n, 2))[avail]
    if np.any(slot_team[chosen] != team_column):
        raise ValueError("nearest_slot names a slot belonging to the other team.")

    counts = f["team_visible_slot_count"]
    _require_dtype("team_visible_slot_count", counts, "int")
    _require_shape("team_visible_slot_count", counts, (n, 2))
    capacity = np.array([np.sum(slot_team == HOME), np.sum(slot_team == AWAY)])
    if np.any(counts < 0) or np.any(counts > capacity):
        raise ValueError("team_visible_slot_count is outside 0..slots-per-team.")
    if np.any(avail & (counts < 1)):
        raise ValueError("geometry_available is set for a team with no visible slot.")
    expected_avail = visible[:, None] & (counts > 0)
    if not np.array_equal(avail, expected_avail):
        raise ValueError(
            "geometry_available must equal ball_visible AND "
            "a positive team_visible_slot_count."
        )
    return f


def geometry_evidence(features):
    """Pure function: Stage 2 feature dict -> geometry evidence dict (see OUTPUT_SCHEMA).

    No threshold, interpolation, velocity or smoothing. Works for any N >= 1.
    """
    f = validate_geometry(features)
    n = f["frame"].shape[0]
    dist = f["nearest_dist_m"].astype(np.float64)
    avail = f["geometry_available"].astype(bool)
    both = avail[:, HOME] & avail[:, AWAY]

    margin = np.full(n, np.nan, dtype=np.float64)
    # Subtract only where both are finite, so inf - inf never happens.
    margin[both] = dist[both, AWAY] - dist[both, HOME]

    nearest_team = np.full(n, NO_TEAM, dtype=np.int8)
    # Comparisons against +inf are well defined; a tie (including inf == inf) stays -1.
    nearest_team[dist[:, HOME] < dist[:, AWAY]] = HOME
    nearest_team[dist[:, AWAY] < dist[:, HOME]] = AWAY

    return {
        "frame": f["frame"].astype(np.int32),
        "ball_visible": f["ball_visible"].astype(bool),
        "geometry_available": avail,
        "nearest_dist_m": dist.astype(np.float32),
        "both_teams_available": both,
        "distance_margin_m": margin.astype(np.float32),
        "nearest_team": nearest_team,
        "nearest_distance_m": np.min(dist, axis=1).astype(np.float32),
        "team_visible_slot_count": f["team_visible_slot_count"].astype(np.int32),
    }


# ---------------------------------------------------------------------------------------------
# 2. Tactical evidence
# ---------------------------------------------------------------------------------------------
def normalise(value):
    """strip() + casefold(); None -> ''."""
    return "" if value is None else str(value).strip().casefold()


def build_team_lookup(team_names):
    """{normalised name: team index} from [home names, away names].

    Each side may be a single string or a sequence of explicit aliases. Every alias is supplied
    by the caller; nothing is inferred, and there is no "everything else is away" rule.
    """
    if isinstance(team_names, (str, bytes)) or len(team_names) != 2:
        raise ValueError("team_names must be [home, away].")
    lookup = {}
    for team, names in enumerate(team_names):
        aliases = [names] if isinstance(names, str) else list(names)
        normalised = [normalise(alias) for alias in aliases]
        if not normalised or any(not alias for alias in normalised):
            raise ValueError("Each team needs at least one non-empty name.")
        for alias in normalised:
            if alias == NA_TEAM:
                raise ValueError("'N/A' cannot be used as a team name.")
            if lookup.get(alias, team) != team:
                raise ValueError("Team name {!r} is assigned to both teams.".format(alias))
            lookup[alias] = team
    return lookup


def _team_tag(tags):
    """The Team tag value, found by case-insensitive group name. None if absent or conflicting."""
    values = {normalise(value) for group, value in (tags or {}).items()
              if normalise(group) == "team"}
    return values.pop() if len(values) == 1 else None


def _frame_int(event, key, index):
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("Event {} has a non-integer {}: {!r}.".format(index, key, value))
    return int(value)


def tactical_evidence(events, n_frames, team_names):
    """Pure function: Smart Tagging events -> (tag evidence dict, summary dict).

    events: the list returned by starter.load_smart_tagging.load(); each item has code,
      start_frame, end_frame (right-exclusive) and tags.
    n_frames: video length N; intervals are clipped to [0, N).
    team_names: [home name(s), away name(s)] as written in the Team tag, supplied by the caller.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, numbers.Integral) or n_frames < 1:
        raise ValueError("n_frames must be a positive integer.")
    n = int(n_frames)
    lookup = build_team_lookup(team_names)

    attacking = np.zeros((n, 2), dtype=bool)
    defensive = np.zeros((n, 2), dtype=bool)
    att_transition = np.zeros((n, 2), dtype=bool)
    def_transition = np.zeros((n, 2), dtype=bool)
    set_piece = np.zeros(n, dtype=bool)
    set_piece_team = np.zeros((n, 2), dtype=bool)
    context_only = np.zeros(n, dtype=bool)
    unhandled = np.zeros(n, dtype=bool)

    summary = {
        "n_events": 0,
        "code_counts": {},
        "unhandled_code_counts": {},
        "team_resolution": {"home": 0, "away": 0, "n/a": 0, "missing": 0, "unknown": 0},
        "unknown_team_values": {},
        "team_events_without_team": 0,
        "empty_intervals": 0,
        "intervals_entirely_outside_video": 0,
        "intervals_clipped": 0,
    }

    for index, event in enumerate(events):
        start = _frame_int(event, "start_frame", index)
        end = _frame_int(event, "end_frame", index)
        if end < start:
            raise ValueError("Event {} is reversed: start_frame {} > end_frame {}."
                             .format(index, start, end))
        summary["n_events"] += 1

        code = normalise(event.get("code"))
        category = EVENT_CODES.get(code)
        if category is None:
            key = code or "<missing>"
            summary["unhandled_code_counts"][key] = summary["unhandled_code_counts"].get(key, 0) + 1
        else:
            summary["code_counts"][code] = summary["code_counts"].get(code, 0) + 1

        team_value = _team_tag(event.get("tags"))
        if not team_value:
            team, resolution = None, "missing"
        elif team_value == NA_TEAM:
            team, resolution = None, "n/a"
        elif team_value in lookup:
            team = lookup[team_value]
            resolution = "home" if team == HOME else "away"
        else:
            team, resolution = None, "unknown"
            summary["unknown_team_values"][team_value] = (
                summary["unknown_team_values"].get(team_value, 0) + 1)
        summary["team_resolution"][resolution] += 1

        if start == end:
            summary["empty_intervals"] += 1
            continue
        lo, hi = max(start, 0), min(end, n)
        if lo >= hi:
            summary["intervals_entirely_outside_video"] += 1
            continue
        if lo != start or hi != end:
            summary["intervals_clipped"] += 1
        span = slice(lo, hi)

        if category is None:
            unhandled[span] = True
        elif category == SET_PIECE:
            set_piece[span] = True
            if team is not None:
                set_piece_team[span, team] = True
        elif category == CONTEXT_ONLY:
            context_only[span] = True
        elif team is None:
            summary["team_events_without_team"] += 1
        elif category == ATTACKING_PHASE:
            attacking[span, team] = True
        elif category == DEFENSIVE_PHASE:
            defensive[span, 1 - team] = True
        elif category == ATTACKING_TRANSITION:
            att_transition[span, team] = True
        elif category == DEFENSIVE_TRANSITION:
            def_transition[span, team] = True

    stable = attacking | defensive
    arrays = {
        "tag_attacking_support": attacking,
        "tag_defensive_support": defensive,
        "tag_stable_support": stable,
        "tag_contradictory_stable_support": stable[:, HOME] & stable[:, AWAY],
        "tag_attacking_transition": att_transition,
        "tag_defensive_transition": def_transition,
        "tag_set_piece": set_piece,
        "tag_set_piece_team": set_piece_team,
        "tag_context_only": context_only,
        "tag_unhandled_code": unhandled,
    }
    summary["frames_true"] = {name: int(array.sum()) for name, array in arrays.items()}
    return arrays, summary


# ---------------------------------------------------------------------------------------------
# 3. Command line (human-run; never triggered by import)
# ---------------------------------------------------------------------------------------------
def _fail(message):
    print("ERROR: " + message, file=sys.stderr)
    return 2


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME, choices=sorted(GAMES))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--features", default=str(here / "local_cache" / "tracking_features.npz"))
    parser.add_argument("--out", default=str(here / "local_cache" / "possession_evidence.npz"))
    parser.add_argument("--home-alias", action="append", default=[],
                        help="extra Smart Tagging Team value for the home side (repeatable)")
    parser.add_argument("--away-alias", action="append", default=[],
                        help="extra Smart Tagging Team value for the away side (repeatable)")
    parser.add_argument("--allow-unmapped-teams", action="store_true",
                        help="continue when a Team value matches neither side (it gets no evidence)")
    args = parser.parse_args(argv)

    features_path = Path(args.features)
    if not features_path.is_file():
        return _fail("Stage 2 cache not found: {}\nRun `python stage2_features.py` first, in a "
                     "normal terminal, or pass --features.".format(features_path))

    with np.load(features_path, allow_pickle=False) as cache:
        features = {key: cache[key] for key in cache.files}
    try:
        geometry = geometry_evidence(features)
    except ValueError as error:
        return _fail("Stage 2 cache failed validation: {}\nRegenerate it with "
                     "`python stage2_features.py`.".format(error))
    n_frames = geometry["frame"].shape[0]
    expected = source_frames("tactical", args.game)
    if n_frames != expected:
        return _fail("Stage 2 cache has {} frames; the tactical video has {}."
                     .format(n_frames, expected))

    meta_path = require(["atd_meta"], args.game, args.data_dir)[0]
    names, slot_teams = read_metadata(meta_path)
    if not np.array_equal(slot_teams, features["slot_team"]):
        return _fail("Tracking metadata slot teams differ from the cache. Regenerate the Stage 2 "
                     "cache with `python stage2_features.py`.")

    team_names = [[names[HOME]] + args.home_alias, [names[AWAY]] + args.away_alias]
    events = load_tags(game=args.game, data_dir=args.data_dir)
    try:
        tags, summary = tactical_evidence(events, n_frames, team_names)
    except ValueError as error:
        return _fail("Smart Tagging failed validation: {}".format(error))

    resolution = summary["team_resolution"]
    if not args.allow_unmapped_teams:
        if summary["unknown_team_values"]:
            return _fail(
                "Smart Tagging Team values match neither tracking team name ({!r} home, {!r} "
                "away): {}\nAdd explicit aliases, e.g. --home-alias \"<value>\" or "
                "--away-alias \"<value>\", or pass --allow-unmapped-teams to give them no "
                "evidence.".format(names[HOME], names[AWAY], summary["unknown_team_values"]))
        if resolution["home"] == 0 or resolution["away"] == 0:
            return _fail("No Smart Tagging event mapped to one side (home {}, away {}). Check "
                         "the team names and aliases.".format(resolution["home"],
                                                              resolution["away"]))

    codes = sorted(EVENT_CODES)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **geometry,
        **tags,
        team_names=np.array(names, dtype=str),
        event_codes=np.array(codes, dtype=str),
        event_categories=np.array([EVENT_CODES[code] for code in codes], dtype=str),
        schema_version=np.int32(SCHEMA_VERSION),
        schema_json=np.array(json.dumps(OUTPUT_SCHEMA)),
        summary_json=np.array(json.dumps(summary, sort_keys=True)),
    )
    print("Teams: home = {}, away = {}".format(*names))
    print("Events: {}  team resolution: {}".format(summary["n_events"], resolution))
    if summary["unhandled_code_counts"]:
        print("Unhandled codes (no evidence assigned): {}".format(summary["unhandled_code_counts"]))
    print("Frames with evidence: {}".format(summary["frames_true"]))
    print("Saved {}. It is a restricted-data derivative: keep it private and Git-ignored."
          .format(out_path))
    print("No possession labels or submission were produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
