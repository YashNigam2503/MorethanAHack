"""Synthetic-fixture tests for possession_evidence.py.

Generated arrays and events only. Nothing here reads data/, local_cache/, predictions or
ground-truth files, and the real match bundle is never touched.

Run only these tests:
    python -m unittest discover -s tests -p "test_possession_evidence.py" -v
"""
import builtins
import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import possession_evidence as pe  # noqa: E402

SLOT_TEAM = np.array([0, 0, 1, 1], dtype=np.int8)
TEAMS = ["Barcelona", "Atletic Club"]


def make_features(rows):
    """rows: [(ball_visible, home_dist or None, away_dist or None)] -> Stage 2-shaped dict."""
    n = len(rows)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    dist = np.full((n, 2), np.inf, dtype=np.float32)
    slot = np.full((n, 2), -1, dtype=np.int32)
    counts = np.zeros((n, 2), dtype=np.int32)
    for i, (visible, home, away) in enumerate(rows):
        if visible:
            ball[i] = (50.0, 30.0)
        for team, value, slot_id in ((0, home, 0), (1, away, 2)):
            if value is not None:
                dist[i, team] = value
                slot[i, team] = slot_id
                counts[i, team] = 1
    return {
        "frame": np.arange(n, dtype=np.int32),
        "ball_xy_m": ball,
        "ball_visible": np.all(np.isfinite(ball), axis=1),
        "nearest_dist_m": dist,
        "nearest_slot": slot,
        "team_visible_slot_count": counts,
        "geometry_available": np.isfinite(dist),
        "slot_team": SLOT_TEAM.copy(),
        "pitch_m": np.array([105.3, 68.0]),
    }


def event(code, start, end, team=None, **extra):
    tags = dict(extra)
    if team is not None:
        tags["Team"] = team
    return {"code": code, "start_s": start / 25.0, "end_s": end / 25.0,
            "start_frame": start, "end_frame": end, "tags": tags}


def frames(mask):
    return np.flatnonzero(mask).tolist()


# --- 1-3: geometry observations -------------------------------------------------------------
class NearestTeamTests(unittest.TestCase):
    def test_home_nearer_away_nearer_and_tie(self):
        g = pe.geometry_evidence(make_features(
            [(True, 1.0, 3.0), (True, 4.0, 2.0), (True, 2.5, 2.5)]))
        self.assertEqual(g["nearest_team"].tolist(), [0, 1, -1])
        np.testing.assert_allclose(g["distance_margin_m"], [2.0, -2.0, 0.0])
        np.testing.assert_allclose(g["nearest_distance_m"], [1.0, 2.0, 2.5])
        self.assertTrue(np.all(g["both_teams_available"]))

    def test_output_dtypes_match_schema(self):
        g = pe.geometry_evidence(make_features([(True, 1.0, 3.0)] * 3))
        for name, array in g.items():
            self.assertIn(name, pe.OUTPUT_SCHEMA)
            self.assertEqual(array.dtype, np.dtype(pe.OUTPUT_SCHEMA[name][1]), msg=name)
            self.assertEqual(array.shape[0], 3, msg=name)


class MissingGeometryTests(unittest.TestCase):
    def test_ball_missing(self):
        g = pe.geometry_evidence(make_features([(False, None, None)]))
        self.assertFalse(g["ball_visible"][0])
        self.assertFalse(np.any(g["geometry_available"]))
        self.assertFalse(g["both_teams_available"][0])
        self.assertTrue(np.isnan(g["distance_margin_m"][0]))
        self.assertEqual(int(g["nearest_team"][0]), -1)
        self.assertEqual(float(g["nearest_distance_m"][0]), np.inf)

    def test_only_one_team_available(self):
        g = pe.geometry_evidence(make_features([(True, 2.0, None), (True, None, 5.0)]))
        self.assertEqual(g["geometry_available"].tolist(), [[True, False], [False, True]])
        self.assertFalse(np.any(g["both_teams_available"]))
        self.assertTrue(np.all(np.isnan(g["distance_margin_m"])))
        # Only the observed team can be nearest; that is still an observation, not a label.
        self.assertEqual(g["nearest_team"].tolist(), [0, 1])
        np.testing.assert_allclose(g["nearest_distance_m"], [2.0, 5.0])


class NonfiniteTests(unittest.TestCase):
    def test_no_invalid_arithmetic_warnings(self):
        rows = [(False, None, None), (True, 2.0, None), (True, None, 1.0), (True, 1.0, 1.0)]
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            g = pe.geometry_evidence(make_features(rows))
        self.assertEqual(np.isnan(g["distance_margin_m"]).tolist(), [True, True, True, False])
        self.assertFalse(np.any(np.isneginf(g["nearest_distance_m"])))

    def test_nan_distance_rejected_not_fabricated(self):
        f = make_features([(True, 1.0, 2.0)])
        f["nearest_dist_m"][0, 1] = np.nan
        f["geometry_available"][0, 1] = False
        with self.assertRaises(ValueError):
            pe.geometry_evidence(f)

    def test_negative_infinity_rejected(self):
        f = make_features([(True, 1.0, None)])
        f["nearest_dist_m"][0, 1] = -np.inf
        with self.assertRaises(ValueError):
            pe.geometry_evidence(f)

    def test_infinite_ball_coordinate_rejected(self):
        f = make_features([(False, None, None)])
        f["ball_xy_m"][0] = (np.inf, np.nan)
        with self.assertRaises(ValueError):
            pe.geometry_evidence(f)


# --- 4: validation --------------------------------------------------------------------------
class ValidationTests(unittest.TestCase):
    def base(self):
        return make_features([(True, 1.0, 2.0), (True, 3.0, None), (False, None, None)])

    def assert_rejects(self, f):
        with self.assertRaises(ValueError):
            pe.geometry_evidence(f)

    def test_valid_fixture_passes(self):
        pe.geometry_evidence(self.base())

    def test_missing_key(self):
        f = self.base()
        del f["nearest_slot"]
        self.assert_rejects(f)

    def test_bad_shapes(self):
        f = self.base()
        f["nearest_dist_m"] = np.ones((3, 3), dtype=np.float32)
        self.assert_rejects(f)
        f = self.base()
        f["ball_visible"] = f["ball_visible"][:2]
        self.assert_rejects(f)
        f = self.base()
        f["pitch_m"] = np.array([105.3])
        self.assert_rejects(f)

    def test_wrong_dtypes(self):
        f = self.base()
        f["geometry_available"] = f["geometry_available"].astype(np.int8)
        self.assert_rejects(f)
        f = self.base()
        f["frame"] = f["frame"].astype(np.float64)
        self.assert_rejects(f)

    def test_frame_gap_duplicate_and_offset(self):
        for bad in ([0, 2, 3], [0, 0, 1], [1, 2, 3], [0, 2, 1]):
            f = self.base()
            f["frame"] = np.array(bad, dtype=np.int32)
            self.assert_rejects(f)

    def test_empty_rejected(self):
        self.assert_rejects(make_features([]))

    def test_mask_says_available_but_distance_infinite(self):
        f = self.base()
        f["geometry_available"][1, 1] = True
        self.assert_rejects(f)

    def test_mask_says_unavailable_but_distance_finite(self):
        f = self.base()
        f["geometry_available"][0, 0] = False
        self.assert_rejects(f)

    def test_negative_distance(self):
        f = self.base()
        f["nearest_dist_m"][0, 0] = -0.5
        self.assert_rejects(f)

    def test_ball_visible_disagrees_with_coordinates(self):
        f = self.base()
        f["ball_visible"][2] = True
        self.assert_rejects(f)

    def test_geometry_without_ball(self):
        f = make_features([(False, None, None)])
        f["nearest_dist_m"][0, 0] = 1.0
        f["geometry_available"][0, 0] = True
        f["nearest_slot"][0, 0] = 0
        f["team_visible_slot_count"][0, 0] = 1
        self.assert_rejects(f)

    def test_nearest_slot_wrong_team_or_out_of_range(self):
        f = self.base()
        f["nearest_slot"][0, 0] = 2  # an away slot in the home column
        self.assert_rejects(f)
        f = self.base()
        f["nearest_slot"][0, 0] = 9
        self.assert_rejects(f)
        f = self.base()
        f["nearest_slot"][2, 0] = 0  # set where geometry is unavailable
        self.assert_rejects(f)

    def test_slot_counts(self):
        f = self.base()
        f["team_visible_slot_count"][0, 0] = 3  # home has only 2 slots
        self.assert_rejects(f)
        f = self.base()
        f["team_visible_slot_count"][0, 0] = 0  # available but no visible slot
        self.assert_rejects(f)
        f = self.base()
        f["team_visible_slot_count"][2, 1] = -1
        self.assert_rejects(f)

    def test_slot_team_values(self):
        f = self.base()
        f["slot_team"] = np.array([0, 0, 1, 2], dtype=np.int8)
        self.assert_rejects(f)

    def test_visible_ball_and_visible_slots_require_geometry(self):
        for team in (pe.HOME, pe.AWAY):
            with self.subTest(team=team):
                f = make_features([(True, 1.0, 2.0)])
                # Keep the visible slot count positive, but hide its geometry.
                f["nearest_dist_m"][0, team] = np.inf
                f["geometry_available"][0, team] = False
                f["nearest_slot"][0, team] = -1
                self.assert_rejects(f)

    def test_missing_ball_with_visible_slots_remains_valid(self):
        f = make_features([(False, None, None)])
        f["team_visible_slot_count"][0] = [1, 1]
        g = pe.geometry_evidence(f)
        self.assertFalse(g["ball_visible"][0])
        self.assertFalse(np.any(g["geometry_available"]))
        self.assertEqual(int(g["nearest_team"][0]), pe.NO_TEAM)
        np.testing.assert_array_equal(
            g["team_visible_slot_count"], [[1, 1]]
        )

    def test_visible_ball_without_visible_slots_remains_valid(self):
        g = pe.geometry_evidence(make_features([(True, None, None)]))
        self.assertTrue(g["ball_visible"][0])
        self.assertFalse(np.any(g["geometry_available"]))
        self.assertEqual(int(g["nearest_team"][0]), pe.NO_TEAM)
        self.assertTrue(np.isposinf(g["nearest_distance_m"][0]))


# --- 5-11: tactical tags --------------------------------------------------------------------
class TeamInterpretationTests(unittest.TestCase):
    def test_attacking_supports_tagged_team_defensive_supports_other(self):
        tags, _ = pe.tactical_evidence(
            [event("BUILD UP", 0, 3, team="Barcelona"),
             event("DEFENDING IN MIDDLE THIRD", 5, 8, team="Barcelona")], 10, TEAMS)
        self.assertEqual(frames(tags["tag_attacking_support"][:, 0]), [0, 1, 2])
        self.assertEqual(frames(tags["tag_attacking_support"][:, 1]), [])
        self.assertEqual(frames(tags["tag_defensive_support"][:, 1]), [5, 6, 7])
        self.assertEqual(frames(tags["tag_defensive_support"][:, 0]), [])
        self.assertEqual(frames(tags["tag_stable_support"][:, 0]), [0, 1, 2])
        self.assertEqual(frames(tags["tag_stable_support"][:, 1]), [5, 6, 7])

    def test_every_defensive_code_flips(self):
        for code in ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                     "DEFENDING IN ATTACKING THIRD"):
            tags, _ = pe.tactical_evidence([event(code, 0, 2, team="Atletic Club")], 4, TEAMS)
            self.assertEqual(frames(tags["tag_defensive_support"][:, 0]), [0, 1], msg=code)


class TeamNameTests(unittest.TestCase):
    def test_case_and_whitespace_variants(self):
        tags, summary = pe.tactical_evidence(
            [event("build up", 0, 1, team="  barcelona "),
             event("  Build Up  ", 1, 2, team="ATLETIC CLUB")], 3, [" BARCELONA", "atletic club "])
        self.assertTrue(tags["tag_attacking_support"][0, 0])
        self.assertTrue(tags["tag_attacking_support"][1, 1])
        self.assertEqual(summary["team_resolution"]["home"], 1)
        self.assertEqual(summary["team_resolution"]["away"], 1)

    def test_aliases(self):
        tags, _ = pe.tactical_evidence([event("BUILD UP", 0, 1, team="Athletic Bilbao")], 2,
                                       [["FC Barcelona", "Barcelona"],
                                        ["Athletic Club", "Athletic Bilbao"]])
        self.assertTrue(tags["tag_attacking_support"][0, 1])

    def test_unknown_team_is_not_assumed_away(self):
        tags, summary = pe.tactical_evidence(
            [event("BUILD UP", 0, 2, team="Real Madrid")], 3, TEAMS)
        self.assertFalse(np.any(tags["tag_attacking_support"]))
        self.assertEqual(summary["unknown_team_values"], {"real madrid": 1})
        self.assertEqual(summary["team_resolution"]["unknown"], 1)
        self.assertEqual(summary["team_events_without_team"], 1)

    def test_na_and_missing_team_give_no_evidence(self):
        events = [event("PROGRESSION", 0, 2, team="N/A"),
                  event("PROGRESSION", 0, 2, team=" n/a "),
                  event("ATTACKING TRANSITION", 0, 2),
                  event("DEFENDING IN MIDDLE THIRD", 0, 2, team="")]
        tags, summary = pe.tactical_evidence(events, 3, TEAMS)
        for name in ("tag_attacking_support", "tag_defensive_support", "tag_stable_support",
                     "tag_attacking_transition", "tag_defensive_transition"):
            self.assertFalse(np.any(tags[name]), msg=name)
        self.assertEqual(summary["team_resolution"]["n/a"], 2)
        self.assertEqual(summary["team_resolution"]["missing"], 2)

    def test_team_group_matched_case_insensitively(self):
        tags, _ = pe.tactical_evidence(
            [{"code": "BUILD UP", "start_frame": 0, "end_frame": 1, "tags": {"team": "Barcelona"}}],
            1, TEAMS)
        self.assertTrue(tags["tag_attacking_support"][0, 0])

    def test_bad_team_mappings_rejected(self):
        for bad in (["Barcelona"], "Barcelona", ["Barcelona", "barcelona "],
                    ["Barcelona", ""], ["Barcelona", "N/A"], [["A", "B"], ["b"]]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                pe.tactical_evidence([], 1, bad)


class SetPieceTests(unittest.TestCase):
    def test_unknown_team_set_piece_still_gives_context(self):
        tags, _ = pe.tactical_evidence(
            [event("SET PIECES", 2, 4, team="N/A", Type="Throw-in"),
             event("SET PIECES", 6, 7, team="Somebody")], 8, TEAMS)
        self.assertEqual(frames(tags["tag_set_piece"]), [2, 3, 6])
        self.assertFalse(np.any(tags["tag_set_piece_team"]))
        self.assertFalse(np.any(tags["tag_stable_support"]))

    def test_known_team_set_piece_is_recorded_not_support(self):
        tags, _ = pe.tactical_evidence(
            [event("SET PIECES", 0, 2, team="Atletic Club", Type="Corner Kick")], 3, TEAMS)
        self.assertEqual(frames(tags["tag_set_piece_team"][:, 1]), [0, 1])
        self.assertFalse(np.any(tags["tag_stable_support"]))
        self.assertFalse(np.any(tags["tag_attacking_transition"]))


class IntervalTests(unittest.TestCase):
    def test_right_exclusive(self):
        tags, _ = pe.tactical_evidence([event("BUILD UP", 2, 5, team="Barcelona")], 10, TEAMS)
        self.assertEqual(frames(tags["tag_attacking_support"][:, 0]), [2, 3, 4])

    def test_clipped_at_both_boundaries(self):
        tags, summary = pe.tactical_evidence(
            [event("BUILD UP", -3, 2, team="Barcelona"),
             event("BUILD UP", 8, 15, team="Atletic Club")], 10, TEAMS)
        self.assertEqual(frames(tags["tag_attacking_support"][:, 0]), [0, 1])
        self.assertEqual(frames(tags["tag_attacking_support"][:, 1]), [8, 9])
        self.assertEqual(summary["intervals_clipped"], 2)

    def test_entirely_outside_contributes_nothing(self):
        tags, summary = pe.tactical_evidence(
            [event("BUILD UP", -5, 0, team="Barcelona"),
             event("SET PIECES", 10, 14),
             event("MYSTERY", 12, 20)], 10, TEAMS)
        for name, array in tags.items():
            self.assertFalse(np.any(array), msg=name)
        self.assertEqual(summary["intervals_entirely_outside_video"], 3)

    def test_empty_interval_contributes_nothing(self):
        tags, summary = pe.tactical_evidence([event("BUILD UP", 4, 4, team="Barcelona")], 10, TEAMS)
        self.assertFalse(np.any(tags["tag_attacking_support"]))
        self.assertEqual(summary["empty_intervals"], 1)

    def test_reversed_rejected(self):
        with self.assertRaises(ValueError):
            pe.tactical_evidence([event("BUILD UP", 5, 4, team="Barcelona")], 10, TEAMS)

    def test_non_integer_frames_rejected(self):
        bad = event("BUILD UP", 0, 2, team="Barcelona")
        bad["end_frame"] = 2.0
        with self.assertRaises(ValueError):
            pe.tactical_evidence([bad], 10, TEAMS)
        bad["end_frame"] = True
        with self.assertRaises(ValueError):
            pe.tactical_evidence([bad], 10, TEAMS)

    def test_numpy_integer_frames_accepted(self):
        ok = event("BUILD UP", np.int64(0), np.int32(2), team="Barcelona")
        tags, _ = pe.tactical_evidence([ok], 3, TEAMS)
        self.assertEqual(frames(tags["tag_attacking_support"][:, 0]), [0, 1])

    def test_bad_frame_count_rejected(self):
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                pe.tactical_evidence([], bad, TEAMS)


class DuplicateAndContradictionTests(unittest.TestCase):
    def test_duplicates_do_not_strengthen(self):
        one, _ = pe.tactical_evidence([event("BUILD UP", 0, 3, team="Barcelona")], 5, TEAMS)
        many, summary = pe.tactical_evidence(
            [event("BUILD UP", 0, 3, team="Barcelona")] * 3
            + [event("PROGRESSION", 1, 3, team="Barcelona")], 5, TEAMS)
        for name in one:
            self.assertEqual(many[name].dtype, np.bool_, msg=name)
            np.testing.assert_array_equal(one[name], many[name], err_msg=name)
        self.assertEqual(summary["code_counts"]["build up"], 3)

    def test_agreeing_attack_and_defence_is_union_not_contradiction(self):
        tags, _ = pe.tactical_evidence(
            [event("BUILD UP", 0, 4, team="Barcelona"),
             event("DEFENDING IN DEFENSIVE THIRD", 2, 6, team="Atletic Club")], 6, TEAMS)
        self.assertEqual(frames(tags["tag_stable_support"][:, 0]), [0, 1, 2, 3, 4, 5])
        self.assertFalse(np.any(tags["tag_contradictory_stable_support"]))

    def test_contradictory_support_flagged_on_overlap_only(self):
        tags, _ = pe.tactical_evidence(
            [event("BUILD UP", 0, 4, team="Barcelona"),
             event("PROGRESSION", 2, 6, team="Atletic Club")], 8, TEAMS)
        self.assertEqual(frames(tags["tag_contradictory_stable_support"]), [2, 3])

    def test_contradiction_via_defensive_tag(self):
        tags, _ = pe.tactical_evidence(
            [event("BUILD UP", 0, 2, team="Barcelona"),
             event("DEFENDING IN MIDDLE THIRD", 0, 2, team="Barcelona")], 2, TEAMS)
        self.assertEqual(frames(tags["tag_contradictory_stable_support"]), [0, 1])


class TransitionTests(unittest.TestCase):
    def test_transition_is_separate_from_stable_support(self):
        tags, _ = pe.tactical_evidence(
            [event("ATTACKING TRANSITION", 0, 3, team="Barcelona"),
             event("DEFENSIVE TRANSITION", 0, 3, team="Atletic Club")], 4, TEAMS)
        self.assertEqual(frames(tags["tag_attacking_transition"][:, 0]), [0, 1, 2])
        self.assertEqual(frames(tags["tag_defensive_transition"][:, 1]), [0, 1, 2])
        self.assertFalse(np.any(tags["tag_attacking_transition"][:, 1]))
        self.assertFalse(np.any(tags["tag_defensive_transition"][:, 0]))
        self.assertFalse(np.any(tags["tag_stable_support"]))

    def test_transition_never_creates_contradiction(self):
        tags, _ = pe.tactical_evidence(
            [event("BUILD UP", 0, 3, team="Barcelona"),
             event("ATTACKING TRANSITION", 0, 3, team="Atletic Club")], 3, TEAMS)
        self.assertFalse(np.any(tags["tag_contradictory_stable_support"]))
        self.assertFalse(np.any(tags["tag_stable_support"][:, 1]))


class UnhandledCodeTests(unittest.TestCase):
    def test_unknown_codes_are_tracked_not_interpreted(self):
        events = [event("COUNTER PRESS", 0, 2, team="Barcelona"),
                  event("SET PIECES EXTRA", 2, 4),
                  event("DEFENDING", 4, 6, team="Barcelona"),   # substring of real codes
                  event(None, 6, 7)]
        tags, summary = pe.tactical_evidence(events, 8, TEAMS)
        self.assertEqual(frames(tags["tag_unhandled_code"]), [0, 1, 2, 3, 4, 5, 6])
        for name, array in tags.items():
            if name != "tag_unhandled_code":
                self.assertFalse(np.any(array), msg=name)
        self.assertEqual(summary["unhandled_code_counts"],
                         {"counter press": 1, "set pieces extra": 1, "defending": 1,
                          "<missing>": 1})

    def test_context_only_codes(self):
        tags, _ = pe.tactical_evidence(
            [event("LONG BALL", 0, 2, team="Barcelona"), event("GOALS", 3, 4, team="Barcelona")],
            5, TEAMS)
        self.assertEqual(frames(tags["tag_context_only"]), [0, 1, 3])
        self.assertFalse(np.any(tags["tag_stable_support"]))
        self.assertFalse(np.any(tags["tag_unhandled_code"]))

    def test_allowlist_is_exactly_the_documented_codes(self):
        documented = {
            "players in the box", "ball in final third", "ball in the box", "progression",
            "defending in middle third", "set pieces", "creating chances",
            "defending in defensive third", "long ball", "attacking transition",
            "defensive transition", "build up", "defending in attacking third", "goals"}
        self.assertEqual(set(pe.EVENT_CODES), documented)


# --- 12: import and CLI safety --------------------------------------------------------------
class ImportSafetyTests(unittest.TestCase):
    def test_import_reads_no_files(self):
        def refuse(*args, **kwargs):
            raise AssertionError("file access during import: {!r}".format(args[:1]))
        with mock.patch.object(builtins, "open", side_effect=refuse), \
                mock.patch.object(np, "load", side_effect=refuse), \
                mock.patch.object(ET, "parse", side_effect=refuse):
            importlib.reload(pe)
        self.assertTrue(callable(pe.main))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pe_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = pe.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_missing_cache_is_actionable_and_reads_nothing_else(self):
        missing = os.path.join(self.tmp, "nope.npz")
        with mock.patch.object(pe, "require", side_effect=AssertionError("data accessed")), \
                mock.patch.object(pe, "load_tags", side_effect=AssertionError("data accessed")):
            code, _, err = self.run_main(["--features", missing])
        self.assertEqual(code, 2)
        self.assertIn("Stage 2 cache not found", err)
        self.assertIn("stage2_features.py", err)

    def write_cache(self, n=4):
        path = os.path.join(self.tmp, "features.npz")
        np.savez_compressed(path, **make_features([(True, 1.0, 2.0)] * n))
        return path

    def start_patches(self, events, n=4, slot_team=SLOT_TEAM,
                      names=("Barcelona", "Atletic Club")):
        for patch in (
                mock.patch.object(pe, "source_frames", return_value=n),
                mock.patch.object(pe, "require", return_value=["synthetic_meta.xml"]),
                mock.patch.object(pe, "read_metadata",
                                  return_value=(list(names), np.array(slot_team, dtype=np.int8))),
                mock.patch.object(pe, "load_tags", return_value=events)):
            patch.start()
            self.addCleanup(patch.stop)

    def test_synthetic_end_to_end_writes_documented_arrays(self):
        out = os.path.join(self.tmp, "evidence.npz")
        self.start_patches([event("BUILD UP", 0, 2, team="Barcelona"),
                            event("PROGRESSION", 2, 4, team="Atletic Club")])
        code, stdout, err = self.run_main(["--features", self.write_cache(), "--out", out])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("No possession labels", stdout)
        with np.load(out, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(pe.OUTPUT_SCHEMA))
            self.assertEqual(saved["team_names"].tolist(), ["Barcelona", "Atletic Club"])
            self.assertEqual(frames(saved["tag_attacking_support"][:, 1]), [2, 3])
            self.assertEqual(json.loads(str(saved["summary_json"]))["n_events"], 2)

    def test_unmapped_team_names_fail_with_alias_hint(self):
        out = os.path.join(self.tmp, "x.npz")
        self.start_patches([event("BUILD UP", 0, 2, team="FC Barcelona"),
                            event("BUILD UP", 2, 4, team="Atletic Club")])
        code, _, err = self.run_main(["--features", self.write_cache(), "--out", out])
        self.assertEqual(code, 2)
        self.assertIn("--home-alias", err)
        self.assertFalse(os.path.exists(out))

    def test_alias_resolves_unmapped_team(self):
        self.start_patches([event("BUILD UP", 0, 2, team="FC Barcelona"),
                            event("BUILD UP", 2, 4, team="Atletic Club")])
        code, _, err = self.run_main(["--features", self.write_cache(),
                                      "--out", os.path.join(self.tmp, "x.npz"),
                                      "--home-alias", "FC Barcelona"])
        self.assertEqual(code, 0, msg=err)

    def test_frame_count_mismatch_fails(self):
        self.start_patches([], n=5)
        code, _, err = self.run_main(["--features", self.write_cache(n=4)])
        self.assertEqual(code, 2)
        self.assertIn("frames", err)

    def test_stale_slot_teams_fail(self):
        self.start_patches([], slot_team=[0, 1, 1, 1])
        code, _, err = self.run_main(["--features", self.write_cache()])
        self.assertEqual(code, 2)
        self.assertIn("Regenerate", err)


if __name__ == "__main__":
    unittest.main()
