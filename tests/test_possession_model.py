"""Synthetic acceptance tests for possession_model.py.

Generated arrays only. Nothing here reads data/, local_cache/, predictions or ground truth.

    python -m unittest discover -s tests -p "test_possession_model.py" -v

Tests use Config(fps=10), so the frame counts are: strong 6, weak 10, tag-only 20, max gap 2,
set-piece stationary 10, outside 2, restart 3, restart-owner window 10.
"""
import builtins
import copy
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import possession_model as pm  # noqa: E402

CFG = pm.Config(fps=10)
H, A = pm.HOME, pm.AWAY
PITCH = (105.3, 68.0)
CENTRE = (50.0, 30.0)


class Scene:
    """Builds internally consistent evidence + tracking dicts frame by frame."""

    def __init__(self, n):
        self.n = n
        self.ball = np.full((n, 2), np.nan)
        self.dist = np.full((n, 2), np.inf)
        self.stable = np.zeros((n, 2), dtype=bool)
        self.set_piece = np.zeros(n, dtype=bool)

    def ball_at(self, frames, xy):
        for i in frames:
            self.ball[i] = xy
        return self

    def geom(self, frames, home=None, away=None):
        """Team distances; a missing ball is placed at CENTRE so geometry stays consistent."""
        for i in frames:
            if np.any(np.isnan(self.ball[i])):
                self.ball[i] = CENTRE
            if home is not None:
                self.dist[i, H] = home
            if away is not None:
                self.dist[i, A] = away
        return self

    def strong(self, frames, team):
        near, far = 1.0, 5.0
        return self.geom(frames, home=near if team == H else far, away=far if team == H else near)

    def tag(self, frames, team):
        for i in frames:
            self.stable[i, team] = True
        return self

    def set_pieces(self, frames):
        self.set_piece[list(frames)] = True
        return self

    def build(self):
        n = self.n
        dist = self.dist.astype(np.float32)
        avail = np.isfinite(dist)
        both = avail.all(axis=1)
        margin = np.full(n, np.nan, dtype=np.float32)
        margin[both] = dist[both, A] - dist[both, H]
        team = np.full(n, -1, dtype=np.int8)
        team[dist[:, H] < dist[:, A]] = H
        team[dist[:, A] < dist[:, H]] = A
        visible = np.all(np.isfinite(self.ball), axis=1)
        evidence = {
            "frame": np.arange(n, dtype=np.int32),
            "ball_visible": visible,
            "geometry_available": avail,
            "nearest_dist_m": dist,
            "both_teams_available": both,
            "distance_margin_m": margin,
            "nearest_team": team,
            "nearest_distance_m": dist.min(axis=1),
            "tag_stable_support": self.stable.copy(),
            "tag_contradictory_stable_support": self.stable.all(axis=1),
            "tag_set_piece": self.set_piece.copy(),
        }
        tracking = {
            "frame": np.arange(n, dtype=np.int32),
            "ball_xy_m": self.ball.astype(np.float32),
            "ball_visible": visible.copy(),
            "pitch_m": np.array(PITCH),
        }
        return evidence, tracking


def run(scene, blocks=None, config=CFG):
    evidence, tracking = scene.build()
    labels = pm.decode(evidence, tracking, blocks or [(1, 0, scene.n - 1)], config)
    return "".join(labels)


def owned_by_home(n):
    """Strong home control on frames 0..5 -> H from frame 5."""
    return Scene(n).strong(range(0, 6), H)


def dead_scene(n=60, kick=25):
    """Home owns; set-piece window 10..39 with a stationary ball -> D at 19; ball kicked from
    `kick` onwards at 2 m per frame."""
    scene = owned_by_home(n).set_pieces(range(10, 40))
    scene.ball_at(range(6, kick), (60.0, 30.0))
    for k, i in enumerate(range(kick, n), start=1):
        scene.ball_at([i], (60.0 + 2.0 * k, 30.0))
    return scene


# --- possession -----------------------------------------------------------------------------
class AcquisitionTests(unittest.TestCase):
    def test_initial_acquisition_on_exact_confirmation_frame(self):
        self.assertEqual(run(Scene(20).strong(range(20), H)), "X" * 5 + "H" * 15)

    def test_confirmed_opposite_team_control(self):
        scene = Scene(30).strong(range(0, 10), H).strong(range(10, 30), A)
        self.assertEqual(run(scene), "X" * 5 + "H" * 10 + "A" * 15)

    def test_deflection_flicker_preserves_possession(self):
        scene = Scene(40).strong(range(40), H)
        scene.strong([10, 13, 16, 19, 25, 26], A)
        self.assertEqual(run(scene), "X" * 5 + "H" * 35)

    def test_loose_and_aerial_ball_preserves_possession(self):
        scene = owned_by_home(40)
        scene.geom(range(6, 15), home=6.0, away=6.5)    # distant ball
        scene.geom(range(15, 25), home=1.5, away=1.5)   # contested tie inside the radius
        scene.geom(range(25, 30), home=1.2, away=1.6)   # close but under the advantage margin
        # 30..39: ball missing (aerial / occluded)
        self.assertEqual(run(scene), "X" * 5 + "H" * 35)


class GapTests(unittest.TestCase):
    def test_short_gap_pauses_without_counting(self):
        scene = owned_by_home(30).strong(range(10, 13), A).strong(range(15, 18), A)
        labels = run(scene)  # 13, 14 missing = max gap (2)
        self.assertEqual(labels[16], "H")
        self.assertEqual(labels[17], "A")    # 6th qualifying frame; gap frames not counted

    def test_long_gap_resets_pending_candidate(self):
        scene = owned_by_home(30).strong(range(10, 13), A).strong(range(16, 22), A)
        labels = run(scene)  # 13..15 missing = 3 > max gap
        self.assertEqual(labels[18], "H")    # would have confirmed here without the reset
        self.assertEqual(labels[20], "H")
        self.assertEqual(labels[21], "A")    # fresh count: 16..21

    def test_resumed_owner_control_clears_candidate(self):
        scene = owned_by_home(30).strong(range(10, 13), A).strong([13], H).strong(range(14, 17), A)
        self.assertNotIn("A", run(scene)[:19])
        scene.strong(range(17, 20), A)
        self.assertEqual(run(scene)[19], "A")  # 14..19 is the new run of six


class OneSidedTests(unittest.TestCase):
    def test_one_sided_close_without_tag_is_not_control(self):
        self.assertEqual(run(Scene(30).geom(range(30), home=1.0)), "X" * 30)

    def test_one_sided_close_with_agreeing_tag_is_weak_route(self):
        scene = Scene(30).geom(range(30), home=1.0).tag(range(30), H)
        self.assertEqual(run(scene), "X" * 9 + "H" * 21)

    def test_missing_opposition_is_not_an_infinite_margin(self):
        scene = owned_by_home(40).geom(range(6, 40), away=0.5)
        self.assertEqual(run(scene), "X" * 5 + "H" * 35)


class TagRouteTests(unittest.TestCase):
    def test_tag_only_fallback_after_longer_confirmation(self):
        self.assertEqual(run(Scene(30).tag(range(30), H)), "X" * 19 + "H" * 11)

    def test_contradictory_tags_give_no_preference(self):
        scene = Scene(40).tag(range(40), H).tag(range(40), A)
        self.assertEqual(run(scene), "X" * 40)

    def test_opposing_strong_geometry_beats_tags(self):
        scene = Scene(40).tag(range(40), A).strong(range(40), H)
        self.assertEqual(run(scene), "X" * 5 + "H" * 35)

    def test_close_opposing_geometry_blocks_tag_only(self):
        scene = Scene(40).tag(range(40), A).geom(range(40), home=1.0)
        self.assertEqual(run(scene), "X" * 40)

    def test_stronger_route_does_not_inherit_weaker_count(self):
        scene = owned_by_home(40).tag(range(6, 21), A).strong(range(21, 40), A)
        labels = run(scene)   # 15 tag-only frames, then strong starts again from 1
        self.assertEqual(labels[25], "H")
        self.assertEqual(labels[26], "A")

    def test_weaker_owner_evidence_does_not_clear_strong_challenger(self):
        scene = owned_by_home(30).strong(range(10, 13), A).tag([13, 14], H)
        scene.strong(range(15, 18), A)
        self.assertEqual(run(scene)[17], "A")

    def test_transition_tags_are_ignored_for_switching(self):
        evidence, tracking = owned_by_home(40).build()
        evidence["tag_attacking_transition"] = np.zeros((40, 2), dtype=bool)
        evidence["tag_attacking_transition"][6:, A] = True
        evidence["tag_defensive_transition"] = np.zeros((40, 2), dtype=bool)
        evidence["tag_defensive_transition"][6:, H] = True
        labels = "".join(pm.decode(evidence, tracking, [(1, 0, 39)], CFG))
        self.assertEqual(labels, "X" * 5 + "H" * 35)

    def test_close_tie_blocks_tag_only_acquisition(self):
        scene = Scene(30).geom(range(30), home=1.0, away=1.0)
        scene.tag(range(30), H)
        self.assertEqual(run(scene), "X" * 30)

    def test_close_tie_does_not_switch_existing_owner(self):
        scene = owned_by_home(40)
        scene.geom(range(6, 40), home=1.0, away=1.0)
        scene.tag(range(6, 40), A)
        self.assertEqual(run(scene), "X" * 5 + "H" * 35)

    def test_distant_tie_still_allows_tag_only_fallback(self):
        scene = Scene(30).geom(range(30), home=4.0, away=4.0)
        scene.tag(range(30), H)
        self.assertEqual(run(scene), "X" * 19 + "H" * 11)


# --- dead ball ------------------------------------------------------------------------------
class DeadBallTests(unittest.TestCase):
    def test_missing_ball_never_produces_dead(self):
        scene = owned_by_home(120).set_pieces(range(50, 80))
        self.assertEqual(run(scene), "X" * 5 + "H" * 115)

    def test_stationary_live_ball_without_set_piece_stays_live(self):
        scene = owned_by_home(60).ball_at(range(6, 60), (60.0, 30.0))
        self.assertEqual(run(scene), "X" * 5 + "H" * 55)

    def test_set_piece_context_alone_is_not_dead(self):
        scene = owned_by_home(50).set_pieces(range(6, 50))
        for i in range(6, 50):
            scene.ball_at([i], (20.0 + (2.0 * i) % 60, 30.0))
        self.assertNotIn("D", run(scene))

    def test_set_piece_plus_stationary_ball_is_dead_on_exact_frame(self):
        scene = owned_by_home(40).ball_at(range(6, 40), (60.0, 30.0)).set_pieces(range(10, 40))
        labels = run(scene)
        self.assertEqual(labels[:19], "X" * 5 + "H" * 14)
        self.assertEqual(labels[19:], "D" * 21)   # 10th stationary set-piece observation

    def test_single_outside_outlier_is_rejected(self):
        scene = owned_by_home(30).ball_at([10], (-5.0, 30.0))
        scene.ball_at([20], (-5.0, 30.0)).ball_at([21], CENTRE).ball_at([22], (-5.0, 30.0))
        self.assertNotIn("D", run(scene))

    def test_sustained_outside_observations_are_dead(self):
        scene = owned_by_home(30).ball_at(range(10, 30), (110.0, 30.0))
        labels = run(scene)
        self.assertEqual(labels[10], "H")
        self.assertEqual(labels[11:], "D" * 19)

    def test_outside_margin_is_respected(self):
        scene = owned_by_home(30).ball_at(range(10, 30), (105.7, 30.0))  # within 0.5 m margin
        self.assertNotIn("D", run(scene))


class RestartTests(unittest.TestCase):
    def test_restart_needs_movement_and_supported_owner(self):
        labels = run(dead_scene().strong(range(20, 25), A))
        self.assertEqual(labels[19:27], "D" * 8)
        self.assertEqual(labels[27], "A")         # 3rd moving observation; owner seen at 24

    def test_restart_without_owner_stays_dead(self):
        self.assertEqual(run(dead_scene())[19:], "D" * 41)

    def test_old_live_team_is_not_reused(self):
        self.assertNotIn("H", run(dead_scene())[19:])

    def test_pre_dead_owner_evidence_is_not_restart_evidence(self):
        scene = dead_scene().strong(range(15, 19), A)   # seen while still live
        self.assertEqual(run(scene)[19:], "D" * 41)

    def test_owner_retained_only_within_window(self):
        inside = dead_scene(kick=25).strong([20, 21], A)
        self.assertEqual(run(inside)[27], "A")              # 27 - 21 = 6 < 10
        stale = dead_scene(kick=35).strong([20, 21], A)
        self.assertEqual(run(stale)[19:], "D" * 41)          # 37 - 21 = 16 >= 10

    def test_missing_frames_after_dead_do_not_restart(self):
        scene = owned_by_home(70).set_pieces(range(10, 40)).ball_at(range(6, 25), (60.0, 30.0))
        scene.strong(range(20, 25), A)        # 25..69: ball missing
        self.assertEqual(run(scene)[19:], "D" * 51)

    def test_no_possession_switch_while_dead(self):
        scene = owned_by_home(70).set_pieces(range(10, 40)).ball_at(range(6, 70), (60.0, 30.0))
        scene.strong(range(20, 70), A)
        self.assertEqual(run(scene)[19:], "D" * 51)

    def test_throw_in_style_restart_from_outside(self):
        scene = owned_by_home(40).ball_at(range(10, 21), (110.0, 30.0))
        scene.strong(range(18, 21), A)
        for k, i in enumerate(range(21, 40)):
            scene.ball_at([i], (104.0 - 2.0 * k, 30.0))
        labels = run(scene)
        self.assertEqual(labels[11:23], "D" * 12)
        self.assertEqual(labels[23], "A")

    def test_outside_stop_allows_continuously_moving_return(self):
        scene = owned_by_home(35)
        scene.ball_at(range(10, 12), (110.0, 30.0))
        for i in range(12, 35):
            scene.ball_at([i], (104.0 - 0.6 * (i - 12), 30.0))
        scene.strong(range(12, 35), A)
        labels = run(scene)
        self.assertEqual(labels[11:14], "D" * 3)
        self.assertEqual(labels[14:], "A" * 21)


# --- optional timing modes ------------------------------------------------------------------
class TimingModeTests(unittest.TestCase):
    """Each backdate_* flag in isolation. Defaults (all off) are pinned by every other test."""

    def cfg(self, **flags):
        return pm.Config(fps=10, **flags)

    def test_defaults_are_off(self):
        self.assertFalse(any(getattr(pm.Config(), name) for name in pm.TIMING_FLAGS))

    def test_backdate_possession_acquisition_and_switch(self):
        cfg = self.cfg(backdate_possession=True)
        self.assertEqual(run(Scene(20).strong(range(20), H), config=cfg), "H" * 20)
        scene = Scene(30).strong(range(0, 10), H).strong(range(10, 30), A)
        self.assertEqual(run(scene, config=cfg), "H" * 10 + "A" * 20)

    def test_backdate_possession_includes_paused_frames_only_within_run(self):
        scene = owned_by_home(30).strong(range(10, 13), A).strong(range(15, 18), A)
        labels = run(scene, config=self.cfg(backdate_possession=True))
        self.assertEqual(labels, "H" * 10 + "A" * 20)   # run starts at 10; 13-14 were paused

    def test_backdate_possession_does_not_revive_rejected_flicker(self):
        scene = Scene(40).strong(range(40), H).strong([10, 13, 16, 19, 25, 26], A)
        self.assertEqual(run(scene, config=self.cfg(backdate_possession=True)), "H" * 40)

    def test_backdate_possession_after_long_gap_uses_fresh_run(self):
        scene = owned_by_home(30).strong(range(10, 13), A).strong(range(16, 22), A)
        labels = run(scene, config=self.cfg(backdate_possession=True))
        self.assertEqual(labels[:16], "H" * 16)          # the reset run 10..12 is not revived
        self.assertEqual(labels[16:], "A" * 14)

    def test_backdate_dead_entry_set_piece(self):
        scene = owned_by_home(40).ball_at(range(6, 40), (60.0, 30.0)).set_pieces(range(10, 40))
        labels = run(scene, config=self.cfg(backdate_dead_entry=True))
        self.assertEqual(labels, "X" * 5 + "H" * 5 + "D" * 30)   # stationary run starts at 10

    def test_backdate_dead_entry_outside(self):
        scene = owned_by_home(30).ball_at(range(10, 30), (110.0, 30.0))
        labels = run(scene, config=self.cfg(backdate_dead_entry=True))
        self.assertEqual(labels, "X" * 5 + "H" * 5 + "D" * 20)

    def test_backdate_dead_entry_keeps_single_outlier_live(self):
        scene = owned_by_home(30).ball_at([10], (-5.0, 30.0))
        self.assertNotIn("D", run(scene, config=self.cfg(backdate_dead_entry=True)))

    def test_backdate_restart(self):
        scene = dead_scene().strong(range(20, 25), A)
        labels = run(scene, config=self.cfg(backdate_restart=True))
        self.assertEqual(labels[19:25], "D" * 6)
        self.assertEqual(labels[25:28], "A" * 3)         # movement run starts at 25
        # (the kicked ball later leaves the pitch, so D returns at 48 in every mode)

    def test_backdate_restart_still_needs_owner(self):
        labels = run(dead_scene(), config=self.cfg(backdate_restart=True))
        self.assertEqual(labels[19:], "D" * 41)

    def test_flags_are_independent(self):
        scene = dead_scene().strong(range(20, 25), A)
        base = run(scene)
        only_restart = run(scene, config=self.cfg(backdate_restart=True))
        differ = [i for i in range(len(base)) if base[i] != only_restart[i]]
        self.assertEqual(differ, [25, 26])

    def test_flags_must_be_booleans(self):
        for name in pm.TIMING_FLAGS:
            with self.assertRaises(ValueError, msg=name):
                pm.Config(**{name: 1})

    def test_default_output_name_never_targets_v1_when_flagged(self):
        self.assertEqual(pm.default_output_name({n: False for n in pm.TIMING_FLAGS}),
                         "submission_v1.csv")
        for name in pm.TIMING_FLAGS:
            flags = {n: n == name for n in pm.TIMING_FLAGS}
            self.assertEqual(pm.default_output_name(flags), "submission_v2.csv", msg=name)


# --- optional dead-from-missing rule --------------------------------------------------------
CALM, RUNNING = (1.0, 2.0), (3.0, 6.0)      # (median, p90) player speed, m/s
DFM = pm.Config(fps=10, dead_from_missing=True)   # minimum calm missing span: 40 frames


def with_motion(scene, speeds):
    """Build a scene and attach player-speed arrays. speeds: {frame: (median, p90)}."""
    evidence, tracking = scene.build()
    median = np.full(scene.n, np.nan, dtype=np.float32)
    p90 = np.full(scene.n, np.nan, dtype=np.float32)
    for i, (m, p) in speeds.items():
        median[i], p90[i] = m, p
    tracking["player_speed_median_mps"], tracking["player_speed_p90_mps"] = median, p90
    return evidence, tracking


def dfm_run(scene, speeds, config=DFM, viterbi=False):
    evidence, tracking = with_motion(scene, speeds)
    decoder = pm.decode_viterbi if viterbi else pm.decode
    return "".join(decoder(evidence, tracking, [(1, 0, scene.n - 1)], config))


def span(frames, value):
    return {i: value for i in frames}


class DeadFromMissingTests(unittest.TestCase):
    """owned_by_home(n): H from frame 5; ball missing from frame 6 unless placed."""

    def test_missing_ball_alone_is_not_enough(self):
        labels = dfm_run(owned_by_home(80), span(range(6, 80), RUNNING))
        self.assertNotIn("D", labels)

    def test_missing_without_player_data_is_not_enough(self):
        self.assertNotIn("D", dfm_run(owned_by_home(80), {}))

    def test_calm_missing_span_confirms_on_exact_frame(self):
        labels = dfm_run(owned_by_home(80), span(range(6, 80), CALM))
        self.assertEqual(labels[:45], "X" * 5 + "H" * 40)
        self.assertEqual(labels[45:], "D" * 35)          # 40th calm missing frame (6..45)

    def test_short_calm_missing_span_stays_live(self):
        scene = owned_by_home(80).ball_at(range(36, 80), CENTRE)   # ball back after 30 frames
        self.assertNotIn("D", dfm_run(scene, span(range(6, 80), CALM)))

    def test_ball_sighting_resets_count(self):
        scene = owned_by_home(90).ball_at([30], CENTRE)
        labels = dfm_run(scene, span(range(6, 90), CALM))
        self.assertEqual(labels[69], "H")                # 6..29 (24 frames) did not qualify
        self.assertEqual(labels.index("D"), 70)          # fresh run 31..70 is the 40th frame

    def test_brief_calm_break_pauses_long_break_resets(self):
        speeds = span(range(6, 90), CALM)
        speeds.update(span([36, 37], RUNNING))           # 2 frames <= max gap: pause
        self.assertEqual(dfm_run(owned_by_home(90), speeds).index("D"), 47)
        speeds.update(span([38], RUNNING))               # 3 frames > max gap: reset
        self.assertEqual(dfm_run(owned_by_home(90), speeds).index("D"), 78)

    def test_flag_off_ignores_motion(self):
        labels = dfm_run(owned_by_home(80), span(range(6, 80), CALM), config=CFG)
        self.assertEqual(labels, run(owned_by_home(80)))
        self.assertNotIn("D", labels)

    def test_existing_rules_still_take_priority(self):
        scene = owned_by_home(40).ball_at(range(10, 40), (110.0, 30.0))
        self.assertEqual(dfm_run(scene, span(range(40), CALM)), run(scene))

    def test_restart_reference_is_first_resighting(self):
        scene = owned_by_home(80)
        for k, i in enumerate(range(60, 80)):            # ball reappears already moving
            scene.ball_at([i], (60.0 + 2.0 * k, 30.0))
        scene.strong(range(60, 63), A)
        labels = dfm_run(scene, span(range(6, 60), CALM))
        self.assertEqual(labels[45:63], "D" * 18)
        self.assertEqual(labels[63], "A")                # 3rd move from the frame-60 sighting

    def test_restart_still_needs_owner(self):
        scene = owned_by_home(80)
        for k, i in enumerate(range(60, 80)):
            scene.ball_at([i], (60.0 + 2.0 * k, 30.0))
        labels = dfm_run(scene, span(range(6, 60), CALM))
        self.assertEqual(labels[45:], "D" * 35)

    def test_composes_with_backdate_dead_entry(self):
        config = pm.Config(fps=10, dead_from_missing=True, backdate_dead_entry=True)
        labels = dfm_run(owned_by_home(80), span(range(6, 80), CALM), config=config)
        self.assertEqual(labels, "X" * 5 + "H" + "D" * 74)

    def test_composes_with_viterbi(self):
        speeds = span(range(6, 80), CALM)
        with_flag = dfm_run(owned_by_home(80), speeds, viterbi=True)
        without = dfm_run(owned_by_home(80), speeds, config=CFG, viterbi=True)
        self.assertEqual(with_flag[6:], "D" * 74)
        self.assertNotIn("D", without)

    def test_viterbi_ignores_short_calm_span(self):
        scene = owned_by_home(80).ball_at(range(36, 80), CENTRE)
        self.assertNotIn("D", dfm_run(scene, span(range(6, 80), CALM), viterbi=True))

    def test_qualifying_runs_match_greedy_counting(self):
        visible = np.zeros(20, dtype=bool)
        visible[10] = True
        quiet = np.ones(20, dtype=bool)
        quiet[[3, 4, 5]] = False                          # 3 > max gap 2: reset
        out = pm._qualifying_quiet(0, 19, visible, quiet, max_gap=2, need=4)
        self.assertEqual(np.flatnonzero(out).tolist(), [6, 7, 8, 9, 11, 12, 13, 14, 15, 16,
                                                        17, 18, 19])

    def test_motion_arrays_validated_only_when_on(self):
        evidence, tracking = owned_by_home(10).build()
        pm.decode(evidence, tracking, [(1, 0, 9)], CFG)        # no motion needed when off
        with self.assertRaises(ValueError):
            pm.decode(evidence, tracking, [(1, 0, 9)], pm.Config(fps=10, dead_from_missing=True))
        evidence, tracking = with_motion(owned_by_home(10), {0: (-1.0, 2.0)})
        with self.assertRaises(ValueError):
            pm.decode(evidence, tracking, [(1, 0, 9)], pm.Config(fps=10, dead_from_missing=True))

    def test_config_validation(self):
        for kwargs in ({"dead_from_missing": 1}, {"missing_dead_s": 0},
                       {"walk_speed_mps": 4.0}, {"run_speed_mps": float("nan")}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                pm.Config(**kwargs)

    def test_output_names_never_collide(self):
        off = {n: False for n in pm.TIMING_FLAGS}
        on = dict(off, backdate_restart=True)
        names = {pm.default_output_name(off), pm.default_output_name(on),
                 pm.default_output_name(off, viterbi=True),
                 pm.default_output_name(off, dead_from_missing=True),
                 pm.default_output_name(off, viterbi=True, dead_from_missing=True),
                 pm.default_output_name(on, dead_from_missing=True)}
        self.assertEqual(names, {"submission_v1.csv", "submission_v2.csv", "submission_v3.csv",
                                 "submission_v4.csv", "submission_v4_viterbi.csv",
                                 "submission_v4_backdate.csv"})


# --- optional Viterbi decoder ---------------------------------------------------------------
SYMMETRIC = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])


def path_letters(emission, transition=SYMMETRIC):
    return "".join(np.array(pm.VITERBI_STATES)[pm._viterbi_path(np.asarray(emission, float),
                                                                    transition)])


def team_emission(burst_cost):
    """10 frames. D excluded (cost 100). H favoured everywhere (A pays 0.1, and 10 at both
    ends); frames 4-5 favour A by `burst_cost` each (H pays it)."""
    e = np.zeros((10, 3))
    e[:, 2] = 100.0
    e[:, 1] = 0.1
    e[[0, 9], 1] = 10.0
    e[[4, 5], 1] = 0.0
    e[[4, 5], 0] = burst_cost
    return e


class ViterbiPathTests(unittest.TestCase):
    """Optimal paths worked out by hand on tiny cost tables."""

    def test_short_burst_does_not_pay_for_two_switches(self):
        # all H = 3+3 = 6 ; H->A->H over 4..5 = 5+5 = 10 ; all A >= 20   -> all H
        self.assertEqual(path_letters(team_emission(3.0)), "H" * 10)

    def test_strong_burst_pays_for_switch_and_return(self):
        # all H = 6+6 = 12 ; H->A->H over exactly 4..5 = 10 (any wider A span adds 0.1/frame)
        self.assertEqual(path_letters(team_emission(6.0)), "HHHHAAHHHH")

    def test_direct_turnover_without_dead_ball(self):
        e = np.zeros((8, 3))
        e[:, 2] = 1.0                 # D slightly disfavoured everywhere
        e[:4, 1] = 4.0                # first half favours H
        e[4:, 0] = 4.0                # second half favours A
        # H*4 A*4 costs 5 ; any route through D costs >= 10 + D emissions
        self.assertEqual(path_letters(e), "HHHHAAAA")

    def test_dead_spell_between_live_spells(self):
        e = np.zeros((9, 3))
        e[:3, 1:] = (9.0, 9.0)        # frames 0-2: H
        e[3:6, :2] = 9.0              # frames 3-5: D (H and A pay 9)
        e[6:, [0, 2]] = 9.0           # frames 6-8: A
        self.assertEqual(path_letters(e), "HHHDDDAAA")

    def test_ties_go_to_lowest_state_and_repeat_identically(self):
        e = np.zeros((6, 3))
        first = path_letters(e)
        self.assertEqual(first, "H" * 6)
        self.assertEqual(path_letters(e), first)

    def test_model_numbers_follow_documented_priors(self):
        transition, weights, initial = pm.viterbi_model(pm.Config())   # 25 fps
        self.assertEqual(initial[0], initial[1])
        self.assertAlmostEqual(initial[2], transition[0, 2] - transition[0, 0])
        p_live, p_dead = 1 / 312.5, 1 / 500.0
        self.assertAlmostEqual(transition[0, 1], -math.log(p_live * 0.5))
        self.assertAlmostEqual(transition[0, 2], -math.log(p_live * 0.5))
        self.assertAlmostEqual(transition[2, 0], -math.log(p_dead * 0.5))
        self.assertAlmostEqual(transition[0, 0], -math.log(1 - p_live))
        self.assertAlmostEqual(transition[2, 2], -math.log(1 - p_dead))
        np.testing.assert_allclose(np.exp(-transition).sum(axis=1), 1.0)
        switch = transition[0, 1] - transition[0, 0]
        self.assertAlmostEqual(weights["strong"] * 15, switch)
        self.assertAlmostEqual(weights["tag_only"] * 50, switch)


class ViterbiDecodeTests(unittest.TestCase):
    def decode(self, scene, blocks=None):
        evidence, tracking = scene.build()
        return "".join(pm.decode_viterbi(evidence, tracking,
                                         blocks or [(1, 0, scene.n - 1)], CFG))

    def test_x_frames_survive_untouched(self):
        scene = Scene(20).strong(range(20), H)
        greedy = run(scene, blocks=[(1, 0, 14)])
        viterbi = self.decode(scene, blocks=[(1, 0, 14)])
        self.assertEqual(greedy, "X" * 5 + "H" * 10 + "X" * 5)
        for g, v in zip(greedy, viterbi):
            self.assertEqual(g == "X", v == "X")
        self.assertEqual(viterbi, "X" * 5 + "H" * 10 + "X" * 5)

    def test_half_boundaries_do_not_carry_state(self):
        scene = Scene(30).strong(range(0, 10), H).strong(range(15, 30), A)
        both = self.decode(scene, blocks=[(1, 0, 9), (2, 15, 29)])
        second_alone = self.decode(scene, blocks=[(2, 15, 29)])
        self.assertEqual(both[15:], second_alone[15:])
        self.assertEqual(both[10:15], "X" * 5)
        self.assertEqual(both, "X" * 5 + "H" * 5 + "X" * 10 + "A" * 10)

    def test_lagged_evidence_never_crosses_half_start(self):
        # A ball that "moves" only across the halftime gap must not create moving evidence.
        scene = Scene(30).ball_at(range(0, 10), (10.0, 30.0)).ball_at(range(15, 30), (90.0, 30.0))
        evidence, tracking = scene.build()
        ev, tr, _, _ = pm.validate_inputs(evidence, tracking, [(1, 0, 9), (2, 15, 29)])
        team, route = pm.observations(ev, CFG)
        _, weights, _ = pm.viterbi_model(CFG)
        cost = pm._viterbi_emissions(15, 29, team, route, tr["ball_xy_m"], tr["ball_visible"],
                                     ev["tag_set_piece"], tr["pitch_m"], CFG, weights)
        self.assertFalse(np.any(cost))

    def test_sustained_outside_ball_becomes_dead(self):
        scene = owned_by_home(40)
        for i in range(6, 20):
            scene.ball_at([i], (20.0 + 2.0 * (i - 6), 30.0))   # live, moving
        scene.ball_at(range(20, 40), (110.0, 30.0))            # out of play
        self.assertEqual(self.decode(scene), "X" * 5 + "H" * 15 + "D" * 20)

    def test_single_outside_outlier_stays_live(self):
        scene = owned_by_home(30).ball_at([15], (-5.0, 30.0))
        self.assertNotIn("D", self.decode(scene))

    def test_deterministic_and_inputs_untouched(self):
        evidence, tracking = dead_scene().strong(range(20, 25), A).build()
        before = copy.deepcopy((evidence, tracking))
        first = pm.decode_viterbi(evidence, tracking, [(1, 0, 59)], CFG)
        second = pm.decode_viterbi(evidence, tracking, [(1, 0, 59)], CFG)
        self.assertEqual(first.tobytes(), second.tobytes())
        for original, now in zip(before, (evidence, tracking)):
            for key in original:
                np.testing.assert_array_equal(original[key], now[key])

    def test_default_decoder_is_not_affected(self):
        scene = dead_scene().strong(range(20, 25), A)
        evidence, tracking = scene.build()
        pm.decode_viterbi(evidence, tracking, [(1, 0, 59)], CFG)
        self.assertEqual(run(scene)[27], "A")      # greedy behaviour pinned elsewhere too

    def test_output_names(self):
        off = {n: False for n in pm.TIMING_FLAGS}
        self.assertEqual(pm.default_output_name(off, viterbi=True), "submission_v3.csv")
        self.assertEqual(pm.default_output_name(off), "submission_v1.csv")


# --- halves and conventions -----------------------------------------------------------------
class HalfTests(unittest.TestCase):
    def test_x_outside_blocks_and_inclusive_last_frame(self):
        labels = run(Scene(20).strong(range(20), H), blocks=[(1, 0, 9)])
        self.assertEqual(labels, "X" * 5 + "H" * 5 + "X" * 10)

    def test_owner_does_not_carry_into_next_half(self):
        labels = run(Scene(30).strong(range(10), H), blocks=[(1, 0, 9), (2, 15, 29)])
        self.assertEqual(labels[9], "H")
        self.assertEqual(labels[15:], "X" * 15)

    def test_pending_count_resets_at_half(self):
        scene = Scene(30).strong(range(6, 10), H).strong(range(15, 30), H)
        labels = run(scene, blocks=[(1, 0, 9), (2, 15, 29)])
        self.assertEqual(labels[:20], "X" * 20)
        self.assertEqual(labels[20:], "H" * 10)

    def test_stationary_history_resets_at_half(self):
        scene = owned_by_home(30).set_pieces(range(30))
        scene.ball_at(range(6, 10), (60.0, 30.0)).ball_at(range(15, 21), (60.0, 30.0))
        scene.strong(range(15, 21), H)
        self.assertNotIn("D", run(scene, blocks=[(1, 0, 9), (2, 15, 29)]))

    def test_unsorted_blocks_are_accepted(self):
        scene = Scene(30).strong(range(30), H)
        self.assertEqual(run(scene, blocks=[(2, 15, 29), (1, 0, 9)]),
                         run(scene, blocks=[(1, 0, 9), (2, 15, 29)]))


class ConfigTests(unittest.TestCase):
    def test_default_frame_counts_use_ceiling(self):
        self.assertEqual(pm.Config().frame_counts(), {
            "strong_confirm": 15, "weak_confirm": 25, "tag_confirm": 50, "max_pending_gap": 5,
            "setpiece_stationary": 25, "out_confirm": 5, "restart_confirm": 8,
            "restart_owner_window": 25})

    def test_ceiling_and_minimum(self):
        self.assertEqual(pm.Config(fps=10).frames(0.25), 3)
        self.assertEqual(pm.Config(fps=10).frames(0.01), 1)

    def test_nonsensical_configs_rejected(self):
        for kwargs in ({"fps": 0}, {"fps": 2.5}, {"fps": True}, {"control_radius_m": 0},
                       {"advantage_m": -1}, {"strong_confirm_s": float("nan")},
                       {"strong_confirm_s": 3.0}, {"tag_confirm_s": 0.8},
                       {"restart_owner_window_s": 0.1}, {"restart_move_m": 0.4},
                       {"out_margin_m": True}, {"stationary_tol_m": "0.5"}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                pm.Config(**kwargs)

    def test_decode_requires_config_instance(self):
        evidence, tracking = Scene(3).build()
        with self.assertRaises(ValueError):
            pm.decode(evidence, tracking, [(1, 0, 2)], {"fps": 10})


class ValidationTests(unittest.TestCase):
    def inputs(self):
        return owned_by_home(10).build()

    def assert_rejects(self, evidence, tracking, blocks=((1, 0, 9),)):
        with self.assertRaises(ValueError):
            pm.decode(evidence, tracking, list(blocks), CFG)

    def test_valid_inputs_pass(self):
        pm.decode(*self.inputs(), [(1, 0, 9)], CFG)

    def test_frame_count_mismatch(self):
        evidence, _ = self.inputs()
        _, tracking = owned_by_home(11).build()
        self.assert_rejects(evidence, tracking)

    def test_frame_order_and_gaps(self):
        for bad in ([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            evidence, tracking = self.inputs()
            evidence["frame"] = np.array(bad, dtype=np.int32)
            self.assert_rejects(evidence, tracking)

    def test_shapes_and_dtypes(self):
        evidence, tracking = self.inputs()
        evidence["tag_stable_support"] = evidence["tag_stable_support"][:, :1]
        self.assert_rejects(evidence, tracking)
        evidence, tracking = self.inputs()
        evidence["tag_set_piece"] = evidence["tag_set_piece"].astype(np.int8)
        self.assert_rejects(evidence, tracking)
        evidence, tracking = self.inputs()
        del evidence["nearest_team"]
        self.assert_rejects(evidence, tracking)

    def test_mask_inconsistencies(self):
        mutations = [
            lambda e, t: e["ball_visible"].__setitem__(8, True),
            lambda e, t: e["both_teams_available"].__setitem__(0, False),
            lambda e, t: e["distance_margin_m"].__setitem__(0, 9.0),
            lambda e, t: e["nearest_team"].__setitem__(0, 1),
            lambda e, t: e["tag_contradictory_stable_support"].__setitem__(0, True),
            lambda e, t: e["geometry_available"].__setitem__((0, 0), False),
            lambda e, t: t["ball_xy_m"].__setitem__((0, 0), np.inf),
        ]
        for index, mutate in enumerate(mutations):
            evidence, tracking = self.inputs()
            mutate(evidence, tracking)
            with self.subTest(index=index):
                self.assert_rejects(evidence, tracking)

    def test_bad_half_blocks(self):
        evidence, tracking = self.inputs()
        for blocks in ([(1, 0, 10)], [(1, -1, 5)], [(1, 5, 4)], [(1, 0, 5), (2, 5, 9)],
                       [(1, 0, 3), (1, 5, 9)], [(1, 0)], [], [(1, 0.0, 5)]):
            with self.subTest(blocks=blocks):
                self.assert_rejects(evidence, tracking, blocks)


class DeterminismTests(unittest.TestCase):
    def test_same_output_and_inputs_untouched(self):
        evidence, tracking = dead_scene().strong(range(20, 25), A).build()
        before = copy.deepcopy((evidence, tracking))
        first = pm.decode(evidence, tracking, [(1, 0, 59)], CFG)
        second = pm.decode(evidence, tracking, [(1, 0, 59)], CFG)
        np.testing.assert_array_equal(first, second)
        for original, now in zip(before, (evidence, tracking)):
            for key in original:
                np.testing.assert_array_equal(original[key], now[key])

    def test_only_valid_labels(self):
        labels = pm.decode(*dead_scene().strong(range(20, 25), A).build(), [(1, 5, 50)], CFG)
        self.assertTrue(set(labels.tolist()) <= set("HADX"))


# --- import and CLI -------------------------------------------------------------------------
class ImportSafetyTests(unittest.TestCase):
    def test_import_reads_no_files(self):
        def refuse(*args, **kwargs):
            raise AssertionError("file access during import: {!r}".format(args[:1]))
        # Load an independent copy so the shared `pm` module (and CFG's class) is untouched.
        spec = importlib.util.spec_from_file_location(
            "possession_model_import_probe", os.path.join(ROOT, "possession_model.py"))
        probe = importlib.util.module_from_spec(spec)
        with mock.patch.object(builtins, "open", side_effect=refuse), \
                mock.patch.object(np, "load", side_effect=refuse):
            spec.loader.exec_module(probe)
        self.assertTrue(callable(probe.main))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="model_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.evidence_path = os.path.join(self.tmp, "possession_evidence.npz")
        self.tracking_path = os.path.join(self.tmp, "tracking_features.npz")
        self.out = os.path.join(self.tmp, "submission_v1.csv")

    def write_inputs(self, n=60, schema=True, pitch=PITCH):
        evidence, tracking = dead_scene(n=n).strong(range(20, 25), A).build()
        if schema:
            evidence["schema_version"] = np.int32(1)
        # String/JSON metadata, as possession_evidence writes it, must be ignored.
        evidence["team_names"] = np.array(["Home FC", "Away FC"])
        evidence["summary_json"] = np.array(json.dumps({"n_events": 3}))
        tracking["pitch_m"] = np.array(pitch)
        np.savez_compressed(self.evidence_path, **evidence)
        np.savez_compressed(self.tracking_path, **tracking)
        return n

    def run_main(self, n, extra=()):
        argv = ["--evidence", self.evidence_path, "--tracking", self.tracking_path,
                "--out", self.out] + list(extra)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(pm, "source_frames", return_value=n), \
                mock.patch.object(pm, "half_video_blocks", return_value=[(1, 0, n - 1)]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = pm.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_synthetic_end_to_end_csv_and_config(self):
        n = self.write_inputs()
        code, stdout, stderr = self.run_main(n)
        self.assertEqual(code, 0, msg=stderr)
        self.assertIn(self.out, stdout)
        with open(self.out, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0], "frame,label")
        self.assertEqual(len(lines), n + 1)
        rows = [line.split(",") for line in lines[1:]]
        self.assertEqual([int(frame) for frame, _ in rows], list(range(n)))
        self.assertTrue({label for _, label in rows} <= set("HADX"))
        with open(os.path.join(self.tmp, "submission_v1_config.json"), encoding="utf-8") as fh:
            record = json.load(fh)
        self.assertEqual(record["config"]["control_radius_m"], 2.0)
        self.assertEqual(record["frame_counts"]["strong_confirm"], 15)  # real fps 25
        self.assertEqual(record["half_blocks_inclusive"], [[1, 0, n - 1]])
        self.assertEqual(sum(record["label_counts"].values()), n)

    def test_timing_flags_are_recorded_in_config(self):
        n = self.write_inputs()
        code, _, stderr = self.run_main(n, extra=["--backdate-restart"])
        self.assertEqual(code, 0, msg=stderr)
        with open(os.path.join(self.tmp, "submission_v1_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)["config"]
        self.assertTrue(config["backdate_restart"])
        self.assertFalse(config["backdate_possession"])
        self.assertFalse(config["backdate_dead_entry"])

    def test_viterbi_cli_writes_and_records_priors(self):
        n = self.write_inputs()
        code, _, stderr = self.run_main(n, extra=["--viterbi"])
        self.assertEqual(code, 0, msg=stderr)
        with open(self.out, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines), n + 1)
        self.assertTrue({line.split(",")[1] for line in lines[1:]} <= set("HADX"))
        with open(os.path.join(self.tmp, "submission_v1_config.json"), encoding="utf-8") as fh:
            record = json.load(fh)
        self.assertEqual(record["viterbi"]["live_mean_s"], 12.5)
        self.assertEqual(len(record["viterbi"]["transition_cost_nats"]), 3)

    def test_default_cli_record_has_no_viterbi_section(self):
        n = self.write_inputs()
        self.assertEqual(self.run_main(n)[0], 0)
        with open(os.path.join(self.tmp, "submission_v1_config.json"), encoding="utf-8") as fh:
            self.assertNotIn("viterbi", json.load(fh))

    def test_viterbi_rejects_backdate_combination(self):
        n = self.write_inputs()
        code, _, stderr = self.run_main(n, extra=["--viterbi", "--backdate-restart"])
        self.assertEqual(code, 2)
        self.assertIn("--viterbi", stderr)
        self.assertFalse(os.path.exists(self.out))

    def write_motion(self, n, median=1.0, p90=2.0, schema=True, frame=None):
        path = os.path.join(self.tmp, "player_motion.npz")
        arrays = {"frame": np.arange(n, dtype=np.int32) if frame is None else frame,
                  "player_speed_median_mps": np.full(n, median, dtype=np.float32),
                  "player_speed_p90_mps": np.full(n, p90, dtype=np.float32),
                  "player_speed_count": np.full(n, 20, dtype=np.int32)}
        if schema:
            arrays["schema_version"] = np.int32(1)
        np.savez_compressed(path, **arrays)
        return path

    def test_dead_from_missing_cli_composes_and_records(self):
        n = self.write_inputs()
        motion = self.write_motion(n)
        for extra in (["--dead-from-missing"], ["--dead-from-missing", "--viterbi"],
                      ["--dead-from-missing", "--backdate-dead-entry"]):
            with self.subTest(extra=extra):
                code, _, stderr = self.run_main(n, extra=extra + ["--motion", motion])
                self.assertEqual(code, 0, msg=stderr)
                with open(os.path.join(self.tmp, "submission_v1_config.json"),
                          encoding="utf-8") as fh:
                    config = json.load(fh)["config"]
                self.assertTrue(config["dead_from_missing"])
                self.assertEqual(config["missing_dead_s"], 4.0)

    def test_default_record_has_no_dead_from_missing_fields(self):
        n = self.write_inputs()
        self.assertEqual(self.run_main(n)[0], 0)
        with open(os.path.join(self.tmp, "submission_v1_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)["config"]
        for key in pm.DFM_FIELDS:
            self.assertNotIn(key, config)

    def test_dead_from_missing_without_motion_file_is_actionable(self):
        n = self.write_inputs()
        code, _, stderr = self.run_main(
            n, extra=["--dead-from-missing", "--motion", os.path.join(self.tmp, "none.npz")])
        self.assertEqual(code, 2)
        self.assertIn("stage2_features.py", stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_malformed_motion_file_fails_cleanly(self):
        n = self.write_inputs()
        for kwargs in ({"schema": False}, {"frame": np.arange(1, n + 1, dtype=np.int32)},
                       {"median": -1.0}):
            with self.subTest(kwargs=kwargs):
                motion = self.write_motion(n, **kwargs)
                code, _, stderr = self.run_main(n, extra=["--dead-from-missing",
                                                          "--motion", motion])
                self.assertEqual(code, 2, msg=stderr)
                self.assertNotIn("Traceback", stderr)
                self.assertFalse(os.path.exists(self.out))

    def test_output_passes_public_validator(self):
        try:
            sys.path.insert(0, os.path.join(ROOT, "evaluation"))
            import possession_metrics as metrics
        except ImportError as error:
            self.skipTest("evaluation dependencies unavailable: {}".format(error))
        n = self.write_inputs()
        self.assertEqual(self.run_main(n)[0], 0)
        self.assertIsNone(metrics.validate_submission(metrics.read_label_csv(self.out), n))

    def test_missing_inputs_are_actionable(self):
        code, _, stderr = self.run_main(60)
        self.assertEqual(code, 2)
        self.assertIn("stage2_features.py", stderr)
        self.write_inputs()
        os.remove(self.evidence_path)
        code, _, stderr = self.run_main(60)
        self.assertEqual(code, 2)
        self.assertIn("possession_evidence.py", stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_wrong_game_frame_count(self):
        self.write_inputs(n=60)
        code, _, stderr = self.run_main(61)
        self.assertEqual(code, 2)
        self.assertIn("Wrong game", stderr)

    def test_wrong_game_pitch(self):
        n = self.write_inputs(pitch=(100.0, 64.0))
        code, _, stderr = self.run_main(n)
        self.assertEqual(code, 2)
        self.assertIn("pitch", stderr)

    def test_missing_schema_version(self):
        n = self.write_inputs(schema=False)
        code, _, stderr = self.run_main(n)
        self.assertEqual(code, 2)
        self.assertIn("schema_version", stderr)

    def test_malformed_npz(self):
        n = self.write_inputs()
        with open(self.evidence_path, "w", encoding="utf-8") as handle:
            handle.write("not an npz")
        code, _, stderr = self.run_main(n)
        self.assertEqual(code, 2)
        self.assertIn("Could not read evidence cache", stderr)

    def test_inconsistent_inputs_fail_cleanly(self):
        n = self.write_inputs()
        with np.load(self.tracking_path) as data:
            tracking = {key: data[key] for key in data.files}
        tracking["ball_visible"] = ~tracking["ball_visible"]
        np.savez_compressed(self.tracking_path, **tracking)
        code, _, stderr = self.run_main(n)
        self.assertEqual(code, 2)
        self.assertIn("Regenerate", stderr)

    def rewrite(self, path, **changes):
        """Replace arrays in a temporary synthetic NPZ fixture."""
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays.update(changes)
        np.savez_compressed(path, **arrays)

    def assert_clean_failure(self, n, *fragments):
        code, _, stderr = self.run_main(n)
        self.assertEqual(code, 2, msg=stderr)
        for fragment in fragments:
            self.assertIn(fragment, stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_malformed_schema_versions(self):
        for bad in (np.array([1, 1], dtype=np.int32), np.array("1"), np.float64(1.0),
                    np.float64(1.9), np.bool_(True)):
            with self.subTest(bad=repr(bad)):
                n = self.write_inputs()
                self.rewrite(self.evidence_path, schema_version=bad)
                self.assert_clean_failure(n, "schema_version", "possession_evidence.py")

    def test_scalar_frame_array(self):
        n = self.write_inputs()
        self.rewrite(self.evidence_path, frame=np.int32(0))
        self.assert_clean_failure(n, "frame array", "possession_evidence.py")

    def test_malformed_pitch(self):
        for bad in (np.array([105.3, 68.0, 1.0]), np.array([[105.3, 68.0]]),
                    np.array([105, 68]), np.array([105.3, np.nan]), np.array([105.3, -68.0])):
            with self.subTest(bad=repr(bad)):
                n = self.write_inputs()
                self.rewrite(self.tracking_path, pitch_m=bad)
                self.assert_clean_failure(n, "pitch_m", "stage2_features.py")


if __name__ == "__main__":
    unittest.main()
