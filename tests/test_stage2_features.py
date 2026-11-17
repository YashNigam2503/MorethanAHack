"""Synthetic-fixture tests for stage2_features.py.

Only tiny temporary XML/text fixtures are used. Nothing here reads data/, caches,
predictions or ground-truth files, and the real match bundle is never touched.

Run only these tests:
    python -m unittest discover -s tests -p "test_stage2_features.py" -v
"""
import os
import sys
import shutil
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import stage2_features as s2  # noqa: E402
from starter import baseline  # noqa: E402
from starter import load_smart_tagging as lst  # noqa: E402
from games import GAMES, GAME  # noqa: E402

PITCH = (105.3, 68.0)
FPS = 25
NAN_ROW = "NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN"

META_XML = """<?xml version="1.0"?>
<root>
  <Team id="HOME1"><Name>Home FC</Name></Team>
  <Team id="AWAY2"><Name>Away FC</Name></Team>
  <Player teamId="HOME1" id="p0"/>
  <Player teamId="HOME1" id="p1"/>
  <Player teamId="AWAY2" id="p2"/>
  <Player teamId="AWAY2" id="p3"/>
</root>
"""
SLOTS = np.array([0, 0, 1, 1], dtype=np.int8)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stage2_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path


class MetadataTests(Base):
    def test_slots_map_to_teams_in_xml_order(self):
        names, slot_teams = s2.read_metadata(self.write("meta.xml", META_XML))
        self.assertEqual(names, ["Home FC", "Away FC"])
        np.testing.assert_array_equal(slot_teams, SLOTS)
        self.assertEqual(slot_teams.dtype, np.int8)

    def test_agrees_with_baseline_team_mapping(self):
        path = self.write("meta.xml", META_XML)
        _, s2_slots = s2.read_metadata(path)
        np.testing.assert_array_equal(s2_slots, baseline.team_of_each_track(path))

    def test_rejects_wrong_team_count(self):
        bad = META_XML.replace('<Team id="AWAY2"><Name>Away FC</Name></Team>', "")
        with self.assertRaises(ValueError):
            s2.read_metadata(self.write("m.xml", bad))

    def test_rejects_duplicate_team_ids(self):
        bad = META_XML.replace("AWAY2", "HOME1")
        with self.assertRaises(ValueError):
            s2.read_metadata(self.write("m.xml", bad))

    def test_rejects_missing_team_name(self):
        bad = META_XML.replace("<Name>Away FC</Name>", "")
        with self.assertRaises(ValueError):
            s2.read_metadata(self.write("m.xml", bad))

    def test_rejects_unknown_player_team(self):
        bad = META_XML.replace('teamId="AWAY2" id="p3"', 'teamId="NOPE" id="p3"')
        with self.assertRaises(ValueError):
            s2.read_metadata(self.write("m.xml", bad))

    def test_rejects_no_players(self):
        bad = "\n".join(line for line in META_XML.splitlines() if "<Player" not in line)
        with self.assertRaises(ValueError):
            s2.read_metadata(self.write("m.xml", bad))


class ParseXYTests(unittest.TestCase):
    def test_finite_pair(self):
        self.assertEqual(s2.parse_xy("0.25,0.75,1.0"), (0.25, 0.75))

    def test_nan_is_unavailable(self):
        self.assertIsNone(s2.parse_xy("NaN,NaN,NaN"))
        self.assertIsNone(s2.parse_xy("NaN,0.5,1"))
        self.assertIsNone(s2.parse_xy("0.5,NaN,1"))

    def test_inf_is_unavailable(self):
        self.assertIsNone(s2.parse_xy("inf,0.5,1"))
        self.assertIsNone(s2.parse_xy("0.5,-inf,1"))

    def test_fewer_than_two_fields_rejected(self):
        with self.assertRaises(ValueError):
            s2.parse_xy("0.5")


class TrackingTests(Base):
    def run_extract(self, lines, n_frames=None, slots=SLOTS):
        path = self.write("raw.txt", "".join(line + "\n" for line in lines))
        return s2.extract_tracking(path, slots, n_frames or len(lines), PITCH)

    def test_ball_converted_to_metres_with_pitch_axes(self):
        # x is along the 105.3 m length, y along the 68 m width.
        feats, _ = self.run_extract(["1:0.1,0.1,1;0.1,0.1,1;0.1,0.1,1;0.1,0.1,1:0.5,0.25,1"])
        np.testing.assert_allclose(feats["ball_xy_m"][0], [0.5 * 105.3, 0.25 * 68.0], rtol=1e-6)
        np.testing.assert_allclose(feats["pitch_m"], PITCH)

    def test_frame_index_is_zero_based_and_ordered(self):
        lines = ["{}:{}".format(i, NAN_ROW) for i in range(1, 4)]
        feats, summary = self.run_extract(lines)
        np.testing.assert_array_equal(feats["frame"], [0, 1, 2])
        self.assertEqual(summary["n_frames"], 3)
        self.assertTrue(summary["tracking_ids_exact"])

    def test_nearest_distance_uses_metric_scaling(self):
        # Ball at normalised (0.5, 0.5).
        # Slot 0 (home) is 0.012 normalised in y -> 0.816 m after scaling.
        # Slot 1 (home) is 0.010 normalised in x -> 1.053 m after scaling.
        # Unscaled, slot 1 (0.010) would beat slot 0 (0.012). Scaled, slot 0 wins.
        feats, _ = self.run_extract(["1:0.5,0.512,1;0.51,0.5,1;0.9,0.9,1;0.9,0.1,1:0.5,0.5,1"])
        self.assertEqual(int(feats["nearest_slot"][0, 0]), 0)
        self.assertAlmostEqual(float(feats["nearest_dist_m"][0, 0]), 0.012 * 68.0, places=4)
        self.assertTrue(bool(feats["geometry_available"][0, 0]))
        self.assertTrue(bool(feats["geometry_available"][0, 1]))

    def test_nearest_is_per_team_not_global(self):
        # Away slot 2 is closest to the ball overall; home must still get its own nearest.
        feats, _ = self.run_extract(["1:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1:0.5,0.5,1"])
        self.assertEqual(int(feats["nearest_slot"][0, 1]), 2)
        self.assertEqual(int(feats["nearest_slot"][0, 0]), 1)

    def test_missing_ball_masks_all_distances(self):
        feats, summary = self.run_extract(
            ["1:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1:NaN,NaN,NaN"])
        self.assertFalse(bool(feats["ball_visible"][0]))
        self.assertTrue(np.all(np.isnan(feats["ball_xy_m"][0])))
        self.assertTrue(np.all(np.isinf(feats["nearest_dist_m"][0])))
        self.assertFalse(np.any(feats["geometry_available"][0]))
        self.assertEqual(int(feats["nearest_slot"][0, 0]), -1)
        self.assertEqual(summary["ball_visible_frames"], 0)
        # Player counts are still recorded even without a ball.
        self.assertEqual(int(feats["team_visible_slot_count"][0, 0]), 2)

    def test_missing_players_are_skipped_not_counted(self):
        # Slot 0 is NaN, slot 1 is inf: neither counts. Slot 2 counts for away.
        feats, _ = self.run_extract(["1:NaN,NaN,NaN;inf,0.3,1;0.51,0.5,1;0.9,0.9,1:0.5,0.5,1"])
        self.assertEqual(int(feats["team_visible_slot_count"][0, 0]), 0)
        self.assertEqual(int(feats["team_visible_slot_count"][0, 1]), 2)
        self.assertEqual(int(feats["nearest_slot"][0, 0]), -1)
        self.assertFalse(bool(feats["geometry_available"][0, 0]))
        self.assertTrue(bool(feats["geometry_available"][0, 1]))

    def test_summary_counts_both_teams_and_any(self):
        lines = [
            "1:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1:0.5,0.5,1",      # both teams, ball
            "2:NaN,NaN,NaN;NaN,NaN,NaN;0.51,0.5,1;0.9,0.9,1:0.5,0.5,1",  # away only, ball
            "3:" + NAN_ROW,                                              # nothing
        ]
        _, summary = self.run_extract(lines)
        self.assertEqual(summary["ball_visible_frames"], 2)
        self.assertEqual(summary["ball_and_any_player_frames"], 2)
        self.assertEqual(summary["both_teams_geometry_frames"], 1)

    def test_outside_unit_square_is_recorded_not_dropped(self):
        feats, summary = self.run_extract(
            ["1:1.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1:-0.1,0.5,1"])
        self.assertEqual(summary["finite_ball_observations_outside_unit_square"], 1)
        self.assertEqual(summary["finite_player_observations_outside_unit_square"], 1)
        self.assertTrue(bool(feats["ball_visible"][0]))

    def test_utf8_bom_is_accepted(self):
        path = os.path.join(self.tmp, "bom.txt")
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            handle.write("1:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1:0.5,0.5,1\n")
        feats, _ = s2.extract_tracking(path, SLOTS, 1, PITCH)
        self.assertEqual(int(feats["frame"][0]), 0)

    def test_trailing_separator_is_tolerated(self):
        feats, _ = self.run_extract(["1:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1;:0.5,0.5,1"])
        self.assertEqual(int(feats["team_visible_slot_count"][0, 1]), 2)

    def test_wrong_slot_count_rejected(self):
        with self.assertRaises(ValueError):
            self.run_extract(["1:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1:0.5,0.5,1"])

    def test_duplicate_frame_id_rejected(self):
        with self.assertRaises(ValueError):
            self.run_extract(["1:" + NAN_ROW, "1:" + NAN_ROW, "3:" + NAN_ROW], n_frames=3)

    def test_interior_gap_rejected(self):
        with self.assertRaises(ValueError):
            self.run_extract(["1:" + NAN_ROW, "3:" + NAN_ROW], n_frames=2)

    def test_ids_must_start_at_one(self):
        with self.assertRaises(ValueError):
            self.run_extract(["0:" + NAN_ROW], n_frames=1)

    def test_too_few_records_rejected(self):
        with self.assertRaises(ValueError):
            self.run_extract(["1:" + NAN_ROW], n_frames=2)

    def test_too_many_records_rejected(self):
        with self.assertRaises(ValueError):
            self.run_extract(["1:" + NAN_ROW, "2:" + NAN_ROW], n_frames=1)

    def test_malformed_record_rejected(self):
        # Only two ':'-separated parts, so it is not a valid frame:players:ball record.
        with self.assertRaises(ValueError):
            self.run_extract(["1:0.2,0.2,1;0.3,0.3,1"])

    def test_parse_speed(self):
        self.assertEqual(s2.parse_speed("0.5,0.5,3.25"), 3.25)
        for bad in ("0.5,0.5", "0.5,0.5,NaN", "0.5,0.5,-1", "0.5,0.5,inf", "0.5,0.5,x"):
            self.assertIsNone(s2.parse_speed(bad), msg=bad)

    def test_motion_summaries_and_unchanged_geometry(self):
        lines = ["1:0.2,0.2,1;0.3,0.3,2;0.51,0.5,3;0.9,0.9,10:0.5,0.5,1",
                 "2:NaN,NaN,NaN;0.3,0.3,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN",
                 "3:0.2,0.2,0.5;NaN,NaN,9;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN"]
        path = self.write("raw.txt", "".join(line + "\n" for line in lines))
        plain, plain_summary = s2.extract_tracking(path, SLOTS, 3, PITCH)
        motion = {}
        with_motion, summary = s2.extract_tracking(path, SLOTS, 3, PITCH, motion_out=motion)
        for key in plain:
            np.testing.assert_array_equal(plain[key], with_motion[key], err_msg=key)
        self.assertEqual(plain_summary, summary)
        np.testing.assert_allclose(motion["player_speed_median_mps"], [2.5, np.nan, 0.5])
        np.testing.assert_allclose(motion["player_speed_p90_mps"][[0, 2]],
                                   [np.percentile([1, 2, 3, 10], 90), 0.5])
        # Speed of a slot whose position is missing is ignored (frame 3, slot 1).
        self.assertEqual(motion["player_speed_count"].tolist(), [4, 0, 1])
        np.testing.assert_array_equal(motion["frame"], [0, 1, 2])

    def test_bad_pitch_rejected(self):
        path = self.write("raw.txt", "1:" + NAN_ROW + "\n")
        for bad in [(0, 68.0), (-1, 68.0), (float("nan"), 68.0), (105.3,)]:
            with self.assertRaises(ValueError):
                s2.extract_tracking(path, SLOTS, 1, bad)


def tag(code, start_frame, end_frame, half=None, typ=None, team=None):
    tags = {}
    if half is not None:
        tags["Half"] = half
    if typ is not None:
        tags["Type"] = typ
    if team is not None:
        tags["Team"] = team
    return {
        "code": code,
        "start_s": start_frame / FPS,
        "end_s": end_frame / FPS,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "tags": tags,
    }


def good_kickoffs():
    return [
        tag("SET PIECES", 11170, 11210, half="1st Half", typ="Kick Off (Start)"),
        tag("SET PIECES", 113730, 113770, half="2nd Half", typ="Kick Off (Start)"),
    ]


class TaggingTests(unittest.TestCase):
    def test_good_tags_pass(self):
        report = s2.audit_tags(good_kickoffs(), n_frames=200000)
        self.assertEqual(report["n_events"], 2)
        self.assertTrue(report["clock_check"]["agrees"])
        self.assertEqual(report["intervals_outside_video"], 0)

    def test_reversed_or_empty_interval_rejected(self):
        for start, end in [(100, 100), (200, 100)]:
            with self.assertRaises(ValueError):
                s2.audit_tags(good_kickoffs() + [tag("X", start, end)], n_frames=200000)

    def test_nonfinite_timestamps_rejected(self):
        events = good_kickoffs()
        events[0]["start_s"] = float("nan")
        with self.assertRaises(ValueError):
            s2.audit_tags(events, n_frames=200000)

    def test_missing_kickoff_rejected(self):
        with self.assertRaises(ValueError):
            s2.audit_tags(good_kickoffs()[:1], n_frames=200000)

    def test_kickoff_must_be_right_exclusive_contained(self):
        # Interval ends exactly on the kickoff frame. Right-exclusive means the kickoff is NOT
        # contained, so the audit must refuse, even though check_time_base is inclusive.
        events = [
            tag("SET PIECES", 11100, 11185, half="1st Half", typ="Kick Off (Start)"),
            tag("SET PIECES", 113730, 113770, half="2nd Half", typ="Kick Off (Start)"),
        ]
        with self.assertRaises(ValueError):
            s2.audit_tags(events, n_frames=200000)

    def test_kickoff_starting_exactly_on_frame_is_contained(self):
        events = [
            tag("SET PIECES", 11185, 11230, half="1st Half", typ="Kick Off (Start)"),
            tag("SET PIECES", 113742, 113790, half="2nd Half", typ="Kick Off (Start)"),
        ]
        self.assertTrue(s2.audit_tags(events, n_frames=200000)["right_exclusive_kickoff_check"])

    def test_intervals_outside_video_counted(self):
        events = good_kickoffs() + [tag("X", -5, 10), tag("Y", 199990, 200010)]
        report = s2.audit_tags(events, n_frames=200000)
        self.assertEqual(report["intervals_outside_video"], 2)

    def test_load_converts_seconds_to_frames_from_xml(self):
        tmp = tempfile.mkdtemp(prefix="stage2_tags_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        folder = os.path.join(tmp, "tactical")
        os.makedirs(folder)
        xml = """<?xml version="1.0"?>
<SportsCode><ALL_INSTANCES>
  <instance><code>SET PIECES</code><start>447.4</start><end>448.4</end>
    <label><group>Half</group><text>1st Half</text></label>
    <label><group>Type</group><text>Kick Off (Start)</text></label></instance>
  <instance><code>BUILD UP</code><start>1.0</start><end>2.0</end>
    <label><group>Team</group><text>Home FC</text></label></instance>
</ALL_INSTANCES></SportsCode>
"""
        with open(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), "w",
                  encoding="utf-8") as handle:
            handle.write(xml)
        events = lst.load(game=GAME, data_dir=tmp)
        self.assertEqual(events[0]["start_frame"], round(447.4 * FPS))  # 11185
        self.assertEqual(events[0]["end_frame"], round(448.4 * FPS))    # 11210
        self.assertEqual(events[0]["tags"]["Half"], "1st Half")
        self.assertEqual(events[1]["tags"]["Team"], "Home FC")
        self.assertTrue(events[0]["start_frame"] <= 11185 < events[0]["end_frame"])


class ConstantsTests(unittest.TestCase):
    def test_public_constants_match_expected_values(self):
        # Reads only the public config table in games.py; no data file is opened.
        self.assertEqual(GAMES[GAME]["pitch"], PITCH)
        self.assertEqual(GAMES[GAME]["fps"], FPS)
        self.assertEqual(GAMES[GAME]["n_video_frames"], 199319)

    def test_kickoff_frames_match_loader(self):
        self.assertEqual(lst.KICKOFF_VIDEO_FRAMES, s2.KICKOFF_VIDEO_FRAMES)


if __name__ == "__main__":
    unittest.main()
