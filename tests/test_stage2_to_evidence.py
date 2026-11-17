"""Contract test: real stage2_features output -> possession_evidence input.

Runs the actual Stage 2 producer functions on tiny synthetic XML/text files, saves the result
exactly as stage2_features.main() does (np.savez_compressed), reloads it the way
possession_evidence.main() does (np.load, allow_pickle=False), and feeds it to the evidence layer.
No real data, cache or ground truth is read.

    python -m unittest discover -s tests -p "test_stage2_to_evidence.py" -v
"""
import io
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

import possession_evidence as pe  # noqa: E402
import stage2_features as s2  # noqa: E402

PITCH = (105.3, 68.0)
META_XML = """<?xml version="1.0"?>
<root>
  <Team id="T1"><Name>Barcelona</Name></Team>
  <Team id="T2"><Name>Atletic Club</Name></Team>
  <Player teamId="T1"/><Player teamId="T1"/><Player teamId="T2"/><Player teamId="T2"/>
</root>
"""
# Frames chosen to hit every geometry case the producer can emit.
RAW_LINES = [
    "1:0.5,0.512,1;0.51,0.5,1;0.9,0.9,1;0.9,0.1,1:0.5,0.5,1",       # both teams, home nearer
    "2:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1:0.5,0.5,1",         # both teams, away nearer
    "3:0.2,0.2,1;0.3,0.3,1;0.51,0.5,1;0.9,0.9,1:NaN,NaN,NaN",       # ball missing
    "4:NaN,NaN,NaN;NaN,NaN,NaN;0.51,0.5,1;0.9,0.9,1:0.5,0.5,1",     # away only
    "5:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN",  # nothing
    # Outside-pitch slots plus an exact tie: 17 m each way along the 68 m axis (exact floats).
    "6:1.2,0.5,1;0.5,0.25,1;0.5,0.75,1;-0.1,0.5,1:0.5,0.5,1",
    "7:0.5,0.5,1;NaN,NaN,NaN;0.9,0.9,1;NaN,NaN,NaN:0.5,0.5,1",      # zero distance
]
EXPECTED_NEAREST = [0, 1, -1, 1, -1, -1, 0]


class Stage2ToEvidenceContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="contract_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.meta = self.write("meta.xml", META_XML)
        self.raw = self.write("raw.txt", "".join(line + "\n" for line in RAW_LINES))
        self.names, self.slot_teams = s2.read_metadata(self.meta)
        features, _ = s2.extract_tracking(self.raw, self.slot_teams, len(RAW_LINES), PITCH)
        self.cache = os.path.join(self.tmp, "tracking_features.npz")
        np.savez_compressed(self.cache, **features)   # exactly as stage2_features.main()
        with np.load(self.cache, allow_pickle=False) as loaded:  # exactly as pe.main()
            self.features = {key: loaded[key] for key in loaded.files}

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

    def test_producer_keys_are_exactly_what_consumer_needs(self):
        self.assertEqual(set(self.features), set(pe.GEOMETRY_KEYS))

    def test_round_tripped_cache_passes_validation(self):
        pe.validate_geometry(self.features)

    def test_evidence_values_from_real_producer(self):
        g = pe.geometry_evidence(self.features)
        self.assertEqual(g["nearest_team"].tolist(), EXPECTED_NEAREST)
        self.assertEqual(g["both_teams_available"].tolist(),
                         [True, True, False, False, False, True, True])
        self.assertEqual(np.isnan(g["distance_margin_m"]).tolist(),
                         [False, False, True, True, True, False, False])
        dist = self.features["nearest_dist_m"].astype(np.float64)
        self.assertAlmostEqual(float(g["distance_margin_m"][0]), dist[0, 1] - dist[0, 0], places=4)
        self.assertGreater(float(g["distance_margin_m"][0]), 0)
        self.assertLess(float(g["distance_margin_m"][1]), 0)
        self.assertEqual(float(g["distance_margin_m"][5]), 0.0)
        self.assertEqual(float(g["nearest_distance_m"][6]), 0.0)
        self.assertEqual(float(g["nearest_distance_m"][2]), np.inf)
        self.assertEqual(g["team_visible_slot_count"][2].tolist(), [2, 2])

    def test_metadata_helper_matches_cached_slot_teams(self):
        # possession_evidence.main() compares these to detect a stale cache.
        np.testing.assert_array_equal(self.slot_teams, self.features["slot_team"])
        self.assertEqual(self.slot_teams.dtype, self.features["slot_team"].dtype)

    def test_single_team_metadata_still_compatible(self):
        meta = self.write("one_team.xml", META_XML.replace('teamId="T2"', 'teamId="T1"'))
        _, slots = s2.read_metadata(meta)
        features, _ = s2.extract_tracking(self.raw, slots, len(RAW_LINES), PITCH)
        g = pe.geometry_evidence(features)
        self.assertFalse(np.any(g["geometry_available"][:, 1]))

    def test_cli_runs_on_real_producer_cache(self):
        out = os.path.join(self.tmp, "possession_evidence.npz")
        events = [{"code": "BUILD UP", "start_s": 0.0, "end_s": 0.08, "start_frame": 0,
                   "end_frame": 2, "tags": {"Team": "Barcelona"}},
                  {"code": "PROGRESSION", "start_s": 0.08, "end_s": 0.16, "start_frame": 2,
                   "end_frame": 4, "tags": {"Team": "Atletic Club"}}]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(pe, "source_frames", return_value=len(RAW_LINES)), \
                mock.patch.object(pe, "require", return_value=[self.meta]), \
                mock.patch.object(pe, "load_tags", return_value=events), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            # read_metadata is NOT patched: the real Stage 2 helper reads the synthetic XML.
            code = pe.main(["--features", self.cache, "--out", out])
        self.assertEqual(code, 0, msg=stderr.getvalue())
        with np.load(out, allow_pickle=False) as saved:
            self.assertEqual(saved["team_names"].tolist(), self.names)
            self.assertEqual(saved["nearest_team"].tolist(), EXPECTED_NEAREST)


if __name__ == "__main__":
    unittest.main()
