"""Stage 4: first causal possession / dead-ball decoder. Writes an H/A/D/X submission CSV.

Inputs (both written by earlier stages, both private and Git-ignored):
    local_cache/possession_evidence.npz   possession_evidence.py
    local_cache/tracking_features.npz     stage2_features.py (ball metres, pitch size)
plus the public half bounds from games.half_video_blocks().

Run (human, in a normal terminal):

    python possession_model.py
    python evaluation/validate_submission.py local_cache/submission_v1.csv

Writes local_cache/submission_v1.csv and local_cache/submission_v1_config.json.

This is a hand-written heuristic decoder, not a learned or tuned model. Every parameter in Config
is a provisional engineering default; none was fitted to, or selected on, any ground truth.

Algorithm (per half, causal, state reset at every half)
-------------------------------------------------------
Per-frame observation, strongest route first (evidence-strength precedence):

  3 strong geometry   both teams visible, nearest team within control_radius_m, and its
                      distance advantage over the other team >= advantage_m.
  2 weak geometry     not strong, a single strictly-nearest team within control_radius_m (weak
                      margin, or only one team visible), AND unique stable tag support agrees.
  1 tag only          no geometry within control_radius_m at all, and unique stable tag
                      support. Any close observation blocks it, including a close tie; a
                      distant tie or a distant ball does not.
  0 none              anything else: no ball, distant ball, ties, conflicts, contradictory tags.

"Unique stable tag support" is tag_stable_support for exactly one team; if
tag_contradictory_stable_support is set, tags give no preference. Transition tags are not used.

Possession (only while live). One pending candidate (team, route, count, gap):
  - an observation for the current owner, at a route >= the pending route, clears the pending
    candidate (resumed control);
  - the same team on the same route adds one qualifying frame;
  - a different team, or the same team on a stronger route, starts a NEW candidate at count 1:
    confirmation is never inherited across teams or routes;
  - a weaker observation, or no observation, pauses the candidate: no count is added and the gap
    counter grows; more than max_pending_gap frames in a row clears it.
  The owner switches on the frame the candidate reaches its route's confirmation count, and from
  then on. By default no backdating (see the optional timing modes below). The delays are
  MINIMUM uninterrupted-run latencies: a switch is
  reported at least (confirm - 1) frames after the first qualifying frame, e.g. 14 frames
  (0.56 s) on strong geometry at 25 fps; paused frames and route resets add to the elapsed
  delay. Until the first confirmation in a half the label is X; afterwards uncertainty keeps the
  last owner.

Live -> dead (checked before possession on each live frame; if both rules fire on the same
frame, the outside rule takes priority and supplies the restart reference):
  1 out of pitch      ball observed beyond the pitch by out_margin_m on out_confirm consecutive
                      observations (brief gaps pause the count, longer gaps reset it).
  2 set piece still   inside a SET PIECES interval, ball observed within stationary_tol_m of an
                      anchor on setpiece_stationary consecutive set-piece observations.
  On D: the pending candidate and the restart candidate are cleared, and the label is D.

Dead -> live (the only rule while dead; ordinary switching is suppressed):
  A restart needs (a) the ball observed inside the pitch (+ margin) more than restart_move_m from
  the restart reference on restart_confirm consecutive observations, and (b) a team with strong
  or weak-geometry control observed DURING the dead spell within restart_owner_window. The
  restart goes to that team. The pre-dead owner and tag_set_piece_team are never used.
  The restart reference is the stationary anchor for a set-piece stop, or the last observed
  outside position for an outside stop; it moves to any later position where the ball stays
  within stationary_tol_m for restart_confirm observations, and is dropped after a long gap.
  This is a heuristic, not a verified referee decision.

Never: interpolate the ball, infer speed across a gap, emit D because tracking is missing, or
treat a whole SET PIECES interval as dead.

Optional timing modes (all off by default; off reproduces v1 exactly)
---------------------------------------------------------------------
Confirmation delays exist to reject flicker, but in v1 they also shift every reported boundary
late by at least (confirm - 1) frames. Each mode below removes that structural lag for one kind
of boundary: when a run is confirmed, the frames of that run are relabelled back to its first
qualifying observation. The span is the decoder's own run, not a constant fitted to any labels.
These modes are not causal (they look back at most one confirmation run).

  backdate_possession   switch / first acquisition starts at the run's first qualifying frame
                        (paused frames inside the run are included).
  backdate_dead_entry   D starts at the first outside observation, or the first stationary
                        set-piece observation, of the confirming stop run.
  backdate_restart      the restart team starts at the first moving observation of the run.

CLI: --backdate-possession / --backdate-dead-entry / --backdate-restart. With any of them the
default output is local_cache/submission_v2.csv, so the v1 file is never overwritten.

Optional dead-from-missing rule (off by default)
-----------------------------------------------
--dead-from-missing adds live->dead rule 3, checked after rules 1 and 2: the ball is NOT
observed and the visible players are calm (median speed <= 7 km/h walking limit, 90th
percentile < 14.4 km/h running threshold, from Stage 2's player_motion.npz) for missing_dead_s
(4 s) of such frames; a sighting resets the count, brief calm breaks (<= max_pending_gap) pause
it. Missing ball alone never qualifies. After such a stop the first re-sighting of the ball is
the restart reference; restarts otherwise follow the unchanged rules. It composes with the
backdate modes (D backdated to the calm run's start) and with --viterbi (each frame of a
qualifying run adds missing_quiet cost to H and A). Output: submission_v4[_viterbi|_backdate].

Optional global decoder (off by default)
----------------------------------------
--viterbi replaces greedy confirmation by a per-half Viterbi decode over H/A/D using the same
evidence (see decode_viterbi and the commented priors above it). X frames are exactly those of
the default decoder. Default output: local_cache/submission_v3.csv. Not combinable with the
backdate modes. Without --viterbi the default decoder runs unchanged.
"""
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from games import GAME, GAMES, half_video_blocks, source_frames

HOME, AWAY, NO_TEAM = 0, 1, -1
TEAM_LABELS = ("H", "A")
NONE, TAG_ONLY, WEAK, STRONG = 0, 1, 2, 3
ROUTE_NAMES = {TAG_ONLY: "tag_only", WEAK: "weak_geometry", STRONG: "strong_geometry"}

EVIDENCE_KEYS = ("frame", "ball_visible", "geometry_available", "nearest_dist_m",
                 "both_teams_available", "distance_margin_m", "nearest_team",
                 "nearest_distance_m", "tag_stable_support", "tag_contradictory_stable_support",
                 "tag_set_piece")
TRACKING_KEYS = ("frame", "ball_xy_m", "ball_visible", "pitch_m")
TIMING_FLAGS = ("backdate_possession", "backdate_dead_entry", "backdate_restart")
BOOL_FIELDS = TIMING_FLAGS + ("dead_from_missing",)
DFM_FIELDS = ("dead_from_missing", "missing_dead_s", "walk_speed_mps", "run_speed_mps")
MOTION_KEYS = ("player_speed_median_mps", "player_speed_p90_mps")


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Provisional v1 defaults. General engineering guesses, not fitted or validated values."""
    fps: int = 25
    control_radius_m: float = 2.0
    advantage_m: float = 1.0
    strong_confirm_s: float = 0.6
    weak_confirm_s: float = 1.0
    tag_confirm_s: float = 2.0
    max_pending_gap_s: float = 0.2
    setpiece_stationary_s: float = 1.0
    stationary_tol_m: float = 0.5
    out_margin_m: float = 0.5
    out_confirm_s: float = 0.2
    restart_move_m: float = 1.0
    restart_confirm_s: float = 0.3
    restart_owner_window_s: float = 1.0
    # Optional timing modes (all off = v1). Each relabels, at confirmation time, the frames of the
    # confirming run back to the run's first qualifying observation. No fitted offset is involved:
    # the span is the decoder's own confirmation run.
    backdate_possession: bool = False
    backdate_dead_entry: bool = False
    backdate_restart: bool = False
    # Optional dead-from-missing rule (off = unchanged). Football basis, not fitted:
    dead_from_missing: bool = False
    missing_dead_s: float = 4.0
    #   Minimum calm, ball-missing span. Live occlusions are short: a long ball or high clearance
    #   is in the air for roughly 2-3 s, a crowded-box occlusion a few seconds. Restarts take
    #   longer: even a quick throw-in needs the ball fetched and the thrower set, typically ~10 s
    #   or more, and goal kicks, corners and free kicks take tens of seconds. 4 s sits above
    #   ordinary aerial phases and well below a typical stoppage.
    walk_speed_mps: float = 1.94
    #   7 km/h, the upper limit of "walking" in the standard sports-science speed zones (walking
    #   < 7, jogging 7-14.4, running 14.4-19.8 km/h). Median visible player at or below it =
    #   players static or walking.
    run_speed_mps: float = 4.0
    #   14.4 km/h, the start of the "running" zone. The 90th-percentile visible player below it =
    #   nobody in a live configuration (no pressing sprint, no run onto a pass).

    def __post_init__(self):
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps <= 0:
            raise ValueError("fps must be a positive integer.")
        for name in BOOL_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise ValueError("{} must be True or False.".format(name))
        for item in fields(self):
            if item.name == "fps" or item.name in BOOL_FIELDS:
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{} must be a number.".format(item.name))
            if not (math.isfinite(value) and value > 0):
                raise ValueError("{} must be finite and positive.".format(item.name))
        if not self.strong_confirm_s <= self.weak_confirm_s <= self.tag_confirm_s:
            raise ValueError("Confirmation must not get faster as evidence gets weaker: need "
                             "strong_confirm_s <= weak_confirm_s <= tag_confirm_s.")
        if self.restart_owner_window_s < self.restart_confirm_s:
            raise ValueError("restart_owner_window_s must be >= restart_confirm_s, or no restart "
                             "could ever be confirmed.")
        if self.restart_move_m <= self.stationary_tol_m:
            raise ValueError("restart_move_m must exceed stationary_tol_m.")
        if self.walk_speed_mps >= self.run_speed_mps:
            raise ValueError("walk_speed_mps must be below run_speed_mps.")

    def frames(self, seconds):
        """Seconds -> frame count, ceil(seconds * fps), at least 1. Rounded to 9 decimals first
        so binary noise cannot add a frame (0.6 * 25 is 15, not 16)."""
        return max(1, math.ceil(round(seconds * self.fps, 9)))

    def frame_counts(self):
        return {
            "strong_confirm": self.frames(self.strong_confirm_s),
            "weak_confirm": self.frames(self.weak_confirm_s),
            "tag_confirm": self.frames(self.tag_confirm_s),
            "max_pending_gap": self.frames(self.max_pending_gap_s),
            "setpiece_stationary": self.frames(self.setpiece_stationary_s),
            "out_confirm": self.frames(self.out_confirm_s),
            "restart_confirm": self.frames(self.restart_confirm_s),
            "restart_owner_window": self.frames(self.restart_owner_window_s),
        }


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _array(source, key, kind, shape):
    _check(key in source, "Missing input array {!r}.".format(key))
    array = np.asarray(source[key])
    _check(array.dtype.kind in {"bool": "b", "int": "iu", "float": "f"}[kind],
           "{} must have a {} dtype, got {}.".format(key, kind, array.dtype))
    _check(array.shape == shape, "{} must have shape {}, got {}.".format(key, shape, array.shape))
    return array


def validate_inputs(evidence, tracking, half_blocks):
    """Check shapes, masks, frame order and shared length. Returns (ev, tr, blocks, n)."""
    _check("frame" in evidence and "frame" in tracking, "Both inputs need a 'frame' array.")
    n = int(np.asarray(evidence["frame"]).shape[0]) if np.ndim(evidence["frame"]) == 1 else -1
    _check(n >= 1, "evidence frame must be a non-empty 1-D array.")

    ev = {
        "frame": _array(evidence, "frame", "int", (n,)),
        "ball_visible": _array(evidence, "ball_visible", "bool", (n,)),
        "geometry_available": _array(evidence, "geometry_available", "bool", (n, 2)),
        "nearest_dist_m": _array(evidence, "nearest_dist_m", "float", (n, 2)).astype(np.float64),
        "both_teams_available": _array(evidence, "both_teams_available", "bool", (n,)),
        "distance_margin_m": _array(evidence, "distance_margin_m", "float", (n,))
        .astype(np.float64),
        "nearest_team": _array(evidence, "nearest_team", "int", (n,)),
        "nearest_distance_m": _array(evidence, "nearest_distance_m", "float", (n,))
        .astype(np.float64),
        "tag_stable_support": _array(evidence, "tag_stable_support", "bool", (n, 2)),
        "tag_contradictory_stable_support":
            _array(evidence, "tag_contradictory_stable_support", "bool", (n,)),
        "tag_set_piece": _array(evidence, "tag_set_piece", "bool", (n,)),
    }
    tr = {
        "frame": _array(tracking, "frame", "int", (n,)),
        "ball_xy_m": _array(tracking, "ball_xy_m", "float", (n, 2)).astype(np.float64),
        "ball_visible": _array(tracking, "ball_visible", "bool", (n,)),
        "pitch_m": _array(tracking, "pitch_m", "float", (2,)).astype(np.float64),
    }

    expected = np.arange(n)
    _check(np.array_equal(ev["frame"], expected), "evidence frame must be exactly 0..N-1.")
    _check(np.array_equal(tr["frame"], expected),
           "tracking frame must be exactly 0..N-1 and match the evidence length.")
    _check(np.all(np.isfinite(tr["pitch_m"])) and np.all(tr["pitch_m"] > 0),
           "pitch_m must be two finite positive values.")
    ball = tr["ball_xy_m"]
    _check(not np.any(np.isinf(ball)), "ball_xy_m must use NaN, not infinity, for missing balls.")
    _check(np.array_equal(tr["ball_visible"], np.all(np.isfinite(ball), axis=1)),
           "tracking ball_visible disagrees with finite ball_xy_m.")
    _check(np.array_equal(ev["ball_visible"], tr["ball_visible"]),
           "evidence and tracking disagree on ball visibility; they are from different runs.")

    dist, avail = ev["nearest_dist_m"], ev["geometry_available"]
    _check(not np.any(np.isnan(dist)), "nearest_dist_m contains NaN.")
    _check(np.array_equal(avail, np.isfinite(dist)),
           "geometry_available disagrees with finite nearest_dist_m.")
    _check(np.all(dist[avail] >= 0) and np.all(dist[~avail] == np.inf),
           "nearest_dist_m must be >= 0 where available and +inf elsewhere.")
    _check(not np.any(avail & ~ev["ball_visible"][:, None]),
           "geometry_available is set where the ball is not visible.")
    both = avail.all(axis=1)
    _check(np.array_equal(ev["both_teams_available"], both),
           "both_teams_available disagrees with geometry_available.")
    _check(np.allclose(ev["nearest_distance_m"], dist.min(axis=1), atol=1e-4),
           "nearest_distance_m disagrees with nearest_dist_m.")
    margin = ev["distance_margin_m"]
    _check(np.array_equal(np.isnan(margin), ~both),
           "distance_margin_m must be finite exactly where both teams are available.")
    _check(np.allclose(margin[both], dist[both, 1] - dist[both, 0], atol=1e-4),
           "distance_margin_m disagrees with nearest_dist_m.")
    expected_team = np.full(n, NO_TEAM)
    expected_team[dist[:, 0] < dist[:, 1]] = HOME
    expected_team[dist[:, 1] < dist[:, 0]] = AWAY
    _check(np.array_equal(ev["nearest_team"], expected_team),
           "nearest_team disagrees with nearest_dist_m.")
    stable = ev["tag_stable_support"]
    _check(np.array_equal(ev["tag_contradictory_stable_support"], stable.all(axis=1)),
           "tag_contradictory_stable_support disagrees with tag_stable_support.")

    blocks = []
    for block in half_blocks:
        _check(len(block) == 3, "Each half block must be (half, first_frame, last_frame).")
        half, first, last = block
        for value in block:
            _check(isinstance(value, (int, np.integer)) and not isinstance(value, bool),
                   "Half block values must be integers: {!r}.".format(block))
        _check(0 <= first <= last < n,
               "Half block {!r} is outside frames 0..{}.".format(block, n - 1))
        blocks.append((int(half), int(first), int(last)))
    _check(bool(blocks), "At least one half block is required.")
    _check(len({b[0] for b in blocks}) == len(blocks), "Half numbers must be unique.")
    ordered = sorted(blocks, key=lambda b: b[1])
    for previous, current in zip(ordered, ordered[1:]):
        _check(previous[2] < current[1], "Half blocks overlap: {} and {}.".format(previous, current))
    return ev, tr, ordered, n


# ---------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------
def observations(ev, config):
    """Per-frame (team, route) under the documented precedence. Pure and vectorised."""
    radius, advantage = config.control_radius_m, config.advantage_m
    stable = ev["tag_stable_support"]
    tag_team = np.full(stable.shape[0], NO_TEAM, dtype=np.int8)
    tag_team[stable[:, HOME] & ~stable[:, AWAY]] = HOME
    tag_team[stable[:, AWAY] & ~stable[:, HOME]] = AWAY
    tag_team[ev["tag_contradictory_stable_support"]] = NO_TEAM

    close = ev["nearest_distance_m"] <= radius
    close_team = np.where(close, ev["nearest_team"], NO_TEAM).astype(np.int8)

    margin = ev["distance_margin_m"]
    both = ev["both_teams_available"]
    with np.errstate(invalid="ignore"):
        strong_home = both & close & (close_team == HOME) & (margin >= advantage)
        strong_away = both & close & (close_team == AWAY) & (-margin >= advantage)
    strong = strong_home | strong_away
    weak = ~strong & (close_team != NO_TEAM) & (close_team == tag_team)
    # ~close, not close_team == NO_TEAM: a tie inside the radius is close geometry, not absence.
    tag_only = ~strong & ~weak & (tag_team != NO_TEAM) & ~close

    team = np.full(stable.shape[0], NO_TEAM, dtype=np.int8)
    route = np.zeros(stable.shape[0], dtype=np.int8)
    team[strong_home], team[strong_away] = HOME, AWAY
    route[strong] = STRONG
    team[weak], route[weak] = close_team[weak], WEAK
    team[tag_only], route[tag_only] = tag_team[tag_only], TAG_ONLY
    return team, route


# ---------------------------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------------------------
def _decode_block(first, last, team, route, ball, visible, set_piece, pitch, config, labels,
                  quiet=None):
    counts = config.frame_counts()
    required = {STRONG: counts["strong_confirm"], WEAK: counts["weak_confirm"],
                TAG_ONLY: counts["tag_confirm"]}
    max_gap = counts["max_pending_gap"]
    length, width = pitch
    margin, tol = config.out_margin_m, config.stationary_tol_m

    def outside(p):
        return p[0] < -margin or p[0] > length + margin or p[1] < -margin or p[1] > width + margin

    live, owner = True, NO_TEAM
    pend_team, pend_route, pend_count, pend_gap = NO_TEAM, NONE, 0, 0
    out_count, out_gap = 0, 0                      # live: outside-pitch run
    sp_anchor, sp_count, sp_gap = None, 0, 0       # live: set-piece stationary run
    st_anchor, st_count = None, 0                  # dead: stationarity tracker
    settled, move_count, dead_gap = None, 0, 0     # dead: settled position, movement run
    cand_team, cand_frame = NO_TEAM, None          # dead: restart owner candidate
    # First frame of each confirming run; used only by the optional backdate_* modes.
    pend_start = out_start = sp_start = move_start = first
    # Optional rule 3 (dead_from_missing): calm, ball-missing run. Inert when quiet is None.
    q_count, q_gap, q_start = 0, 0, first
    miss_frames = config.frames(config.missing_dead_s)
    ref_on_reappear = False        # D from missing: first re-sighting becomes restart reference

    for i in range(first, last + 1):
        seen = bool(visible[i])
        p = ball[i] if seen else None

        if live:
            # Rule 1: persistent outside-pitch observations.
            if seen:
                out_gap = 0
                if outside(p):
                    if out_count == 0:
                        out_start = i
                    out_count += 1
                else:
                    out_count = 0
            else:
                out_gap += 1
                if out_gap > max_gap:
                    out_count = 0
            # Rule 2: stationary ball inside a set-piece window.
            if not set_piece[i]:
                sp_anchor, sp_count, sp_gap = None, 0, 0
            elif seen:
                sp_gap = 0
                if sp_anchor is not None and np.hypot(*(p - sp_anchor)) <= tol:
                    sp_count += 1
                else:
                    sp_anchor, sp_count, sp_start = p, 1, i
            else:
                sp_gap += 1
                if sp_gap > max_gap:
                    sp_anchor, sp_count = None, 0
            # Rule 3 (optional): ball missing AND players calm, sustained. Calm-breaking frames
            # inside a missing span pause the count briefly; any ball sighting resets it.
            if quiet is not None:
                if seen:
                    q_count, q_gap = 0, 0
                elif quiet[i]:
                    if q_count == 0:
                        q_start = i
                    q_count, q_gap = q_count + 1, 0
                else:
                    q_gap += 1
                    if q_gap > max_gap:
                        q_count, q_gap = 0, 0

            out_stop = out_count >= counts["out_confirm"]
            sp_stop = sp_count >= counts["setpiece_stationary"]
            miss_stop = quiet is not None and q_count >= miss_frames
            if out_stop or sp_stop or miss_stop:
                live = False
                pend_team, pend_route, pend_count, pend_gap = NO_TEAM, NONE, 0, 0
                cand_team, cand_frame = NO_TEAM, None
                ref_on_reappear = False
                if out_stop:
                    # Confirmation occurs on an observed outside-ball frame.
                    # This is a restart reference, not an assertion of stationarity.
                    settled = p.copy()
                    st_anchor, st_count = None, 0
                    stop_start = out_start
                elif sp_stop:
                    settled = sp_anchor.copy()
                    st_anchor, st_count = sp_anchor.copy(), sp_count
                    stop_start = sp_start
                else:
                    # Ball unseen: no position yet. The first re-sighting becomes the reference.
                    settled, st_anchor, st_count = None, None, 0
                    ref_on_reappear = True
                    stop_start = q_start
                move_count, dead_gap = 0, 0
                if config.backdate_dead_entry:
                    labels[stop_start:i] = "D"
                q_count, q_gap = 0, 0
                out_count = out_gap = sp_count = sp_gap = 0
                sp_anchor = None
                labels[i] = "D"
                continue

            # Possession update.
            t, r = int(team[i]), int(route[i])
            if r == NONE or (pend_team != NO_TEAM and r < pend_route):
                if pend_team != NO_TEAM:
                    pend_gap += 1
                    if pend_gap > max_gap:
                        pend_team, pend_route, pend_count, pend_gap = NO_TEAM, NONE, 0, 0
            elif t == owner:
                pend_team, pend_route, pend_count, pend_gap = NO_TEAM, NONE, 0, 0
            elif t == pend_team and r == pend_route:
                pend_count, pend_gap = pend_count + 1, 0
            else:
                pend_team, pend_route, pend_count, pend_gap = t, r, 1, 0
                pend_start = i
            if pend_team != NO_TEAM and pend_count >= required[pend_route]:
                owner = pend_team
                if config.backdate_possession:
                    labels[pend_start:i] = TEAM_LABELS[owner]
                pend_team, pend_route, pend_count, pend_gap = NO_TEAM, NONE, 0, 0
            labels[i] = TEAM_LABELS[owner] if owner != NO_TEAM else "X"
            continue

        # Dead: only the restart rule runs; ordinary switching is suppressed.
        if route[i] >= WEAK:
            cand_team, cand_frame = int(team[i]), i
        if seen:
            dead_gap = 0
            if settled is None and ref_on_reappear:
                settled = p.copy()
            moving = (settled is not None and np.hypot(*(p - settled)) > config.restart_move_m
                      and not outside(p))
            if moving:
                if move_count == 0:
                    move_start = i
                move_count += 1
            else:
                move_count = 0
            if st_anchor is not None and np.hypot(*(p - st_anchor)) <= tol:
                st_count += 1
            else:
                st_anchor, st_count = p, 1
            if st_count >= counts["restart_confirm"] and not moving:
                settled = st_anchor
        else:
            dead_gap += 1
            if dead_gap > max_gap:
                # No history across an occlusion: the ball may have been anywhere meanwhile.
                st_anchor, st_count, settled, move_count = None, 0, None, 0

        owner_ok = (cand_team != NO_TEAM and cand_frame is not None
                    and i - cand_frame < counts["restart_owner_window"])
        if move_count >= counts["restart_confirm"] and owner_ok:
            live, owner = True, cand_team
            if config.backdate_restart:
                labels[move_start:i] = TEAM_LABELS[owner]
            pend_team, pend_route, pend_count, pend_gap = NO_TEAM, NONE, 0, 0
            out_count = out_gap = sp_count = sp_gap = 0
            sp_anchor, st_anchor, st_count, settled = None, None, 0, None
            move_count, dead_gap, cand_team, cand_frame = 0, 0, NO_TEAM, None
            q_count, q_gap, ref_on_reappear = 0, 0, False
            labels[i] = TEAM_LABELS[owner]
        else:
            labels[i] = "D"


def decode(evidence, tracking, half_blocks, config):
    """Return a length-N array of H/A/D/X labels from synthetic-compatible inputs.

    evidence: possession_evidence arrays (see EVIDENCE_KEYS); tracking: stage2 arrays
    (TRACKING_KEYS); half_blocks: inclusive (half, first_frame, last_frame) video-frame blocks;
    config: Config. X outside the blocks. Inputs are not modified.
    """
    if not isinstance(config, Config):
        raise ValueError("config must be a Config instance.")
    ev, tr, blocks, n = validate_inputs(evidence, tracking, half_blocks)
    team, route = observations(ev, config)
    quiet = quiet_missing(tracking, tr["ball_visible"], config)
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        _decode_block(first, last, team, route, tr["ball_xy_m"], tr["ball_visible"],
                      ev["tag_set_piece"], tr["pitch_m"], config, labels, quiet)
    return labels


def quiet_missing(tracking, visible, config):
    """dead_from_missing corroboration per frame, or None when the rule is off.

    True where the ball is NOT observed AND the visible players are calm: median speed at or
    below walk_speed_mps and 90th-percentile speed below run_speed_mps (Stage 2's
    player_motion arrays). Frames with no valid player speed are never calm.
    """
    if not config.dead_from_missing:
        return None
    n = visible.shape[0]
    arrays = {}
    for key in MOTION_KEYS:
        arrays[key] = _array(tracking, key, "float", (n,)).astype(np.float64)
        _check(not np.any(np.isinf(arrays[key])) and not np.any(arrays[key] < 0),
               "{} must be non-negative, with NaN for no players.".format(key))
    with np.errstate(invalid="ignore"):
        calm = ((arrays["player_speed_median_mps"] <= config.walk_speed_mps)
                & (arrays["player_speed_p90_mps"] < config.run_speed_mps))
    return ~visible & calm


def _qualifying_quiet(first, last, visible, quiet, max_gap, need):
    """Quiet frames belonging to calm missing-ball runs that reach `need` quiet frames, using
    exactly the greedy rule-3 counting (sighting resets; > max_gap calm breaks reset)."""
    out = np.zeros(last - first + 1, dtype=bool)
    run, gap = [], 0
    for k, i in enumerate(range(first, last + 1)):
        if visible[i] or (not quiet[i] and gap + 1 > max_gap):
            if len(run) >= need:
                out[run] = True
            run, gap = [], 0
        elif quiet[i]:
            run.append(k)
            gap = 0
        else:
            gap += 1
    if len(run) >= need:
        out[run] = True
    return out


# ---------------------------------------------------------------------------------------------
# Optional global decoder: Viterbi over H/A/D (CLI --viterbi; off by default)
# ---------------------------------------------------------------------------------------------
# Same evidence, different decoding. The greedy decoder above accepts a change once a run of
# evidence reaches a confirmation count; Viterbi instead finds, per half, the single H/A/D path
# with the lowest total cost = transition costs + per-frame evidence costs. Costs are negative
# natural-log probabilities (nats). Nothing below is estimated from any data.
#
# Football priors (the only numbers introduced here):
VITERBI_LIVE_MEAN_S = 12.5
#   Expected duration of one live possession: the 10-15 s range given in the design brief, taken
#   at its midpoint. Sets P(leave a live state per frame) = 1 / (12.5 s * fps), 1/312.5 at 25 fps.
#   Over 10..15 s the per-frame leave cost would range 5.52..5.93 nats instead of 5.74.
VITERBI_DEAD_MEAN_S = 20.0
#   Expected duration of one dead-ball spell. Effective playing time in top-flight football is
#   commonly reported at roughly 55-60 of ~95 elapsed minutes, i.e. ~35-40 dead minutes, spread
#   over well over 100 restarts per match (throw-ins, free kicks, goal kicks, corners, kick-offs):
#   roughly 20 s per stoppage. Sets P(leave D per frame) = 1 / (20 s * fps), 1/500 at 25 fps.
VITERBI_TURNOVER_SHARE = 0.5
#   Share of live possessions that end by an in-play turnover (H<->A directly) rather than by the
#   ball going dead (H/A->D). No reliable published figure is assumed: 0.5 is the uninformative
#   choice. Turnovers without a stoppage are allowed, as the brief requires.
VITERBI_RESTART_SHARE = 0.5
#   P(a restart goes to H) = P(to A): either team can take a restart; no side is preferred.
# Initial state: each half starts LIVE, as in the default decoder (live=True at every half
# start). H and A start at equal cost; starting in D costs exactly one live->D transition, so D
# still needs positive dead-ball evidence at a half start. (A uniform start would let a half sit
# in D for free: team evidence deliberately does not penalise D, and D's self-cost is lower.)
# Ties between equal-cost paths are broken deterministically towards the lower state index
# (H < A < D) by numpy argmin, so repeat runs are byte-identical.
#
# Emission costs re-use the existing evidence rules and the existing confirmation counts, so no
# new feature or threshold is introduced. Each weight is "one change's transition cost divided by
# the number of frames the greedy decoder needs to confirm that change", so a sustained change is
# accepted after about the same amount of evidence as before, while an isolated burst must pay
# for leaving AND returning, i.e. needs about twice as much.
#   team observation (strong / weak / tag-only route, from observations()):
#       + w_route on the OTHER team.  D is not penalised: players stand next to a dead ball.
#   ball observed outside the pitch by out_margin_m:            + w_out on H and A
#   inside a SET PIECES window, ball within stationary_tol_m of its position restart_confirm
#   frames earlier (both observed):                             + w_still on H and A
#   ball inside the pitch, more than restart_move_m from its position restart_confirm frames
#   earlier (both observed):                                    + w_move on D
#   no ball observation, or none of the above:                  no cost (state carries over)
# The stationary / moving tests are the greedy restart and set-piece rules written per frame
# (fixed lag restart_confirm instead of a stateful anchor), using the same thresholds.
VITERBI_STATES = ("H", "A", "D")
_DEAD = 2


def viterbi_model(config):
    """(transition 3x3 from-row to-column, emission weights, initial costs), all in nats."""
    counts = config.frame_counts()
    leave_live = 1.0 / (VITERBI_LIVE_MEAN_S * config.fps)
    leave_dead = 1.0 / (VITERBI_DEAD_MEAN_S * config.fps)
    stay_live, stay_dead = -math.log1p(-leave_live), -math.log1p(-leave_dead)
    turnover = -math.log(leave_live * VITERBI_TURNOVER_SHARE)
    to_dead = -math.log(leave_live * (1.0 - VITERBI_TURNOVER_SHARE))
    restart = -math.log(leave_dead * VITERBI_RESTART_SHARE)
    transition = np.array([[stay_live, turnover, to_dead],
                           [turnover, stay_live, to_dead],
                           [restart, restart, stay_dead]], dtype=np.float64)
    switch_extra = turnover - stay_live
    stop_extra = to_dead - stay_live
    restart_extra = restart - stay_dead
    weights = {
        "strong": switch_extra / counts["strong_confirm"],
        "weak": switch_extra / counts["weak_confirm"],
        "tag_only": switch_extra / counts["tag_confirm"],
        "outside": stop_extra / counts["out_confirm"],
        "still_set_piece": stop_extra / counts["setpiece_stationary"],
        "moving": restart_extra / counts["restart_confirm"],
        # Used only with dead_from_missing: one stop's cost spread over the minimum calm span.
        "missing_quiet": stop_extra / config.frames(config.missing_dead_s),
    }
    initial = np.array([0.0, 0.0, stop_extra])
    return transition, weights, initial


def _viterbi_emissions(first, last, team, route, ball, visible, set_piece, pitch, config,
                       weights, quiet=None):
    """(frames, 3) evidence costs for one half. Lagged comparisons never cross the half start."""
    lag = config.frame_counts()["restart_confirm"]
    length, width = pitch
    margin = config.out_margin_m
    idx = np.arange(first, last + 1)
    cost = np.zeros((idx.size, 3), dtype=np.float64)

    route_weight = np.zeros(idx.size)
    for code, name in ((STRONG, "strong"), (WEAK, "weak"), (TAG_ONLY, "tag_only")):
        route_weight[route[idx] == code] = weights[name]
    observed_team = team[idx]
    cost[observed_team == HOME, AWAY] += route_weight[observed_team == HOME]
    cost[observed_team == AWAY, HOME] += route_weight[observed_team == AWAY]

    seen = visible[idx]
    p = ball[idx]
    with np.errstate(invalid="ignore"):
        outside = seen & ((p[:, 0] < -margin) | (p[:, 0] > length + margin)
                          | (p[:, 1] < -margin) | (p[:, 1] > width + margin))
    cost[outside, :_DEAD] += weights["outside"]

    has_lag = np.zeros(idx.size, dtype=bool)
    displacement = np.full(idx.size, np.nan)
    if idx.size > lag:
        now, then = np.arange(lag, idx.size), np.arange(0, idx.size - lag)
        has_lag[now] = seen[now] & seen[then]
        delta = p[now] - p[then]
        displacement[now] = np.hypot(delta[:, 0], delta[:, 1])
    with np.errstate(invalid="ignore"):
        still = has_lag & (displacement <= config.stationary_tol_m) & set_piece[idx]
        moving = has_lag & (displacement > config.restart_move_m) & ~outside
    cost[still, :_DEAD] += weights["still_set_piece"]
    cost[moving, _DEAD] += weights["moving"]
    if quiet is not None:
        qualifying = _qualifying_quiet(first, last, visible, quiet,
                                       config.frame_counts()["max_pending_gap"],
                                       config.frames(config.missing_dead_s))
        cost[qualifying, :_DEAD] += weights["missing_quiet"]
    return cost


def _viterbi_path(emission, transition, initial=None):
    """Lowest-cost state path. argmin takes the first minimum, so ties go H < A < D."""
    frames = emission.shape[0]
    back = np.zeros((frames, 3), dtype=np.int8)
    total = emission[0] + (np.zeros(3) if initial is None else initial)
    columns = np.arange(3)
    for t in range(1, frames):
        candidate = total[:, None] + transition
        back[t] = np.argmin(candidate, axis=0)
        total = candidate[back[t], columns] + emission[t]
    path = np.zeros(frames, dtype=np.int8)
    path[-1] = int(np.argmin(total))
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_viterbi(evidence, tracking, half_blocks, config):
    """Viterbi H/A/D decoding per half; X exactly where the default decoder gives X.

    The default decoder is run first and its X frames (outside the half blocks, and before the
    first supported team in each half) are kept untouched; Viterbi labels every other frame.
    """
    greedy = decode(evidence, tracking, half_blocks, config)   # also validates every input
    ev, tr, blocks, _ = validate_inputs(evidence, tracking, half_blocks)
    team, route = observations(ev, config)
    quiet = quiet_missing(tracking, tr["ball_visible"], config)
    transition, weights, initial = viterbi_model(config)
    labels = greedy.copy()
    states = np.array(VITERBI_STATES)
    for _, first, last in blocks:
        emission = _viterbi_emissions(first, last, team, route, tr["ball_xy_m"],
                                      tr["ball_visible"], ev["tag_set_piece"], tr["pitch_m"],
                                      config, weights, quiet)
        block = states[_viterbi_path(emission, transition, initial)]
        keep_x = greedy[first:last + 1] == "X"
        labels[first:last + 1] = np.where(keep_x, "X", block)
    return labels


# ---------------------------------------------------------------------------------------------
# Command line (human-run; never triggered by import)
# ---------------------------------------------------------------------------------------------
def _fail(message):
    print("ERROR: " + message, file=sys.stderr)
    return 2


def _load_npz(path, keys, what, optional=()):
    """Named arrays only. String/JSON metadata is never read as a frame array."""
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in keys if key not in data.files]
            if missing:
                raise ValueError("missing arrays {}".format(missing))
            arrays = {key: data[key] for key in keys}
            extras = {key: data[key] for key in optional if key in data.files}
            return arrays, extras
    except (OSError, ValueError) as error:
        raise ValueError("Could not read {} at {}: {}".format(what, path, error))


def default_output_name(timing, viterbi=False, dead_from_missing=False):
    """v1 default, v2 any timing mode, v3 --viterbi, v4[_viterbi|_backdate] with
    --dead-from-missing. No two flag combinations share a file."""
    if dead_from_missing:
        suffix = "_viterbi" if viterbi else "_backdate" if any(timing.values()) else ""
        return "submission_v4{}.csv".format(suffix)
    if viterbi:
        return "submission_v3.csv"
    return "submission_v2.csv" if any(timing.values()) else "submission_v1.csv"


def write_submission(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(i, label) for i, label in enumerate(labels)))
    return path


def main(argv=None):
    here = Path(__file__).resolve().parent
    cache = here / "local_cache"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME, choices=sorted(GAMES))
    parser.add_argument("--evidence", default=str(cache / "possession_evidence.npz"))
    parser.add_argument("--tracking", default=str(cache / "tracking_features.npz"))
    parser.add_argument("--out", default=None,
                        help="default: local_cache/submission_v1.csv, or submission_v2.csv when "
                             "any --backdate-* mode is on (v1 is never overwritten by default)")
    parser.add_argument("--config-out", default=None,
                        help="default: <out stem>_config.json next to the CSV")
    parser.add_argument("--backdate-possession", action="store_true",
                        help="report a confirmed switch from its first qualifying frame")
    parser.add_argument("--backdate-dead-entry", action="store_true",
                        help="report D from the first frame of the confirming stop run")
    parser.add_argument("--backdate-restart", action="store_true",
                        help="report a restart from the first frame of the movement run")
    parser.add_argument("--viterbi", action="store_true",
                        help="global per-half Viterbi decoding over H/A/D; writes "
                             "submission_v3.csv by default")
    parser.add_argument("--dead-from-missing", action="store_true",
                        help="also treat a sustained ball-missing span with calm players as "
                             "dead-ball evidence (needs local_cache/player_motion.npz)")
    parser.add_argument("--motion", default=str(cache / "player_motion.npz"),
                        help="player-speed summaries written by stage2_features.py")
    args = parser.parse_args(argv)
    timing = {name: getattr(args, name) for name in TIMING_FLAGS}
    if args.viterbi and any(timing.values()):
        return _fail("--viterbi decodes globally and cannot be combined with --backdate-* modes.")
    if args.out is None:
        args.out = str(cache / default_output_name(timing, args.viterbi, args.dead_from_missing))

    for path, producer in ((args.tracking, "stage2_features.py"),
                           (args.evidence, "possession_evidence.py")):
        if not Path(path).is_file():
            return _fail("Input not found: {}\nRun `python {}` first, in a normal terminal, or "
                         "pass the path explicitly.".format(path, producer))
    try:
        evidence, ev_meta = _load_npz(args.evidence, EVIDENCE_KEYS, "evidence cache",
                                      optional=("schema_version",))
        tracking, _ = _load_npz(args.tracking, TRACKING_KEYS, "tracking cache")
    except ValueError as error:
        return _fail(str(error))

    # Metadata guards run before any conversion, so a malformed cache fails cleanly here.
    version = ev_meta.get("schema_version")
    if version is None:
        return _fail("Evidence cache has no schema_version. Re-run "
                     "`python possession_evidence.py`.")
    version = np.asarray(version)
    if version.ndim != 0 or version.dtype.kind not in "iu" or int(version) != 1:
        return _fail("Evidence cache schema_version {!r} (dtype {}, shape {}) is not supported; "
                     "expected the integer scalar 1. Re-run `python possession_evidence.py`."
                     .format(version.tolist(), version.dtype, version.shape))
    frame = np.asarray(evidence["frame"])
    if frame.ndim != 1 or frame.size == 0 or frame.dtype.kind not in "iu":
        return _fail("Evidence frame array must be a non-empty 1-D integer array (got shape {}, "
                     "dtype {}). Re-run `python possession_evidence.py`."
                     .format(frame.shape, frame.dtype))
    pitch = np.asarray(tracking["pitch_m"])
    if (pitch.shape != (2,) or pitch.dtype.kind != "f" or not np.all(np.isfinite(pitch))
            or not np.all(pitch > 0)):
        return _fail("Tracking pitch_m must be two finite positive floats (got shape {}, dtype "
                     "{}). Re-run `python stage2_features.py`.".format(pitch.shape, pitch.dtype))
    n = int(frame.shape[0])
    expected = source_frames("tactical", args.game)
    if n != expected:
        return _fail("Inputs have {} frames but {} has {} tactical video frames. Wrong game or "
                     "stale cache.".format(n, args.game, expected))
    if not np.allclose(pitch, GAMES[args.game]["pitch"]):
        return _fail("Tracking cache pitch {} does not match {} pitch {}. Wrong game or stale "
                     "cache.".format(tracking["pitch_m"].tolist(), args.game,
                                     list(GAMES[args.game]["pitch"])))

    if args.dead_from_missing:
        if not Path(args.motion).is_file():
            return _fail("Input not found: {}\n--dead-from-missing needs player speeds. Re-run "
                         "`python stage2_features.py` (it now also writes player_motion.npz), "
                         "or pass --motion.".format(args.motion))
        try:
            motion, motion_meta = _load_npz(args.motion, ("frame",) + MOTION_KEYS,
                                            "player motion", optional=("schema_version",))
        except ValueError as error:
            return _fail(str(error))
        motion_version = np.asarray(motion_meta.get("schema_version", -1))
        motion_frame = np.asarray(motion["frame"])
        if (motion_version.ndim != 0 or motion_version.dtype.kind not in "iu"
                or int(motion_version) != 1 or motion_frame.ndim != 1
                or motion_frame.dtype.kind not in "iu"
                or not np.array_equal(motion_frame, np.arange(n))):
            return _fail("player_motion.npz is malformed or from a different run (schema 1, "
                         "frames 0..{} expected). Re-run `python stage2_features.py`."
                         .format(n - 1))
        tracking = dict(tracking, **{key: motion[key] for key in MOTION_KEYS})

    config = Config(fps=GAMES[args.game]["fps"], dead_from_missing=args.dead_from_missing,
                    **timing)
    blocks = half_video_blocks(args.game)
    try:
        decoder = decode_viterbi if args.viterbi else decode
        labels = decoder(evidence, tracking, blocks, config)
    except ValueError as error:
        return _fail("Inputs failed validation: {}\nRegenerate them with "
                     "`python stage2_features.py` then `python possession_evidence.py`."
                     .format(error))

    out_path = write_submission(args.out, labels)
    config_path = Path(args.config_out) if args.config_out else \
        out_path.with_name(out_path.stem + "_config.json")
    record = {
        "model": "possession_model v1 (causal heuristic decoder, untuned)",
        "game": args.game,
        # The dead-from-missing fields are recorded only when that rule is on, so records of the
        # earlier versions are unchanged.
        "config": {key: value for key, value in asdict(config).items()
                   if config.dead_from_missing or key not in DFM_FIELDS},
        "frame_counts": config.frame_counts(),
        "frame_convention": "ceil(seconds * fps), minimum 1",
        "half_blocks_inclusive": [list(block) for block in blocks],
        "inputs": {"evidence": Path(args.evidence).name, "tracking": Path(args.tracking).name},
        "label_counts": {label: int(np.sum(labels == label)) for label in "HADX"},
        "note": "Parameters are provisional defaults, not selected on any ground truth.",
    }
    if args.viterbi:   # added only for --viterbi, so the default record is unchanged
        transition, weights, initial = viterbi_model(config)
        record["model"] = "possession_model v3 (per-half Viterbi over H/A/D, untuned priors)"
        record["viterbi"] = {
            "live_mean_s": VITERBI_LIVE_MEAN_S, "dead_mean_s": VITERBI_DEAD_MEAN_S,
            "turnover_share": VITERBI_TURNOVER_SHARE, "restart_share": VITERBI_RESTART_SHARE,
            "states": list(VITERBI_STATES), "transition_cost_nats": transition.tolist(),
            "emission_weight_nats": weights, "initial_cost_nats": initial.tolist(),
        }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print("Wrote {} ({} rows) and {}.".format(out_path, n, config_path))
    print("Private, Git-ignored outputs. Check the format with:\n"
          "  python evaluation/validate_submission.py {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
