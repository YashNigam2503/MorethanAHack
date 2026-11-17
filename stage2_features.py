"""Stage 2: validate tracking and cache geometry. No possession labels or GT access.

Save this file as stage2_features.py at the organiser repository root.
Run: python stage2_features.py
Requires only the existing numpy dependency and public starter modules.
Generated local_cache/ contains restricted-data derivatives: never publish it.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from games import GAME, GAMES, require, source_frames
from starter.load_smart_tagging import load as load_tags
from starter.load_smart_tagging import check_time_base, KICKOFF_VIDEO_FRAMES


def read_metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("Tracking metadata must contain exactly two teams.")
    ids = [team.get("id") for team in teams]
    if None in ids or len(set(ids)) != 2:
        raise ValueError("Tracking team IDs must be present and distinct.")
    names = [team.findtext("Name") for team in teams]
    if any(not name for name in names):
        raise ValueError("Missing tracking team name.")
    slot_teams = []
    for player in root.iter("Player"):
        team_id = player.get("teamId")
        if team_id not in ids:
            raise ValueError("A tracking slot has an unknown team assignment.")
        slot_teams.append(ids.index(team_id))
    if not slot_teams:
        raise ValueError("Tracking metadata contains no player slots.")
    return names, np.asarray(slot_teams, dtype=np.int8)


def parse_xy(group):
    # Speed is deliberately unused. A missing x OR y means unavailable geometry.
    fields = group.strip().split(",")
    if len(fields) < 2:
        raise ValueError("Tracking observation has fewer than two coordinate fields.")
    x, y = float(fields[0]), float(fields[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def parse_speed(group):
    """The documented per-slot speed field (m/s, third field), or None if absent/invalid."""
    fields = group.strip().split(",")
    if len(fields) < 3:
        return None
    try:
        speed = float(fields[2])
    except ValueError:
        return None
    return speed if np.isfinite(speed) and speed >= 0 else None


def extract_tracking(raw_path, slot_teams, n_frames, pitch, motion_out=None):
    """Geometry features and summary, as before.

    If `motion_out` is a dict it is also filled with per-frame player-speed summaries over the
    slots whose position is observed (NaN where no slot has a valid speed):
      player_speed_median_mps, player_speed_p90_mps (float32), player_speed_count (int32).
    The geometry outputs are identical whether or not it is requested.
    """
    scale = np.asarray(pitch, dtype=np.float64)
    want_motion = motion_out is not None
    if want_motion:
        speed_median = np.full(n_frames, np.nan, dtype=np.float32)
        speed_p90 = np.full(n_frames, np.nan, dtype=np.float32)
        speed_count = np.zeros(n_frames, dtype=np.int32)
    if scale.shape != (2,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Pitch dimensions must be two positive metre values.")
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    nearest_dist = np.full((n_frames, 2), np.inf, dtype=np.float32)
    nearest_slot = np.full((n_frames, 2), -1, dtype=np.int32)
    slot_counts = np.zeros((n_frames, 2), dtype=np.int32)
    outside_ball = outside_player = 0
    count = 0
    with open(raw_path, encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError("Malformed tracking record at line {}.".format(line_no))
            frame_id = int(parts[0])
            # Unlike an endpoint-only check, this catches duplicates and interior gaps.
            if frame_id != line_no or line_no > n_frames:
                raise ValueError("Tracking IDs are not exactly 1..N in file order.")
            frame = frame_id - 1
            groups = parts[1].split(";")
            # Accept an optional terminal separator, but never shift slot indices.
            if len(groups) == len(slot_teams) + 1 and not groups[-1].strip():
                groups.pop()
            if len(groups) != len(slot_teams):
                raise ValueError("Player-slot count disagrees with metadata at line {}.".format(line_no))
            ball = parse_xy(parts[2])
            if ball is not None:
                ball_m = np.asarray(ball) * scale
                ball_xy[frame] = ball_m
                outside_ball += int(any(v < 0 or v > 1 for v in ball))
            else:
                ball_m = None
            speeds = [] if want_motion else None
            for slot, group in enumerate(groups):
                xy = parse_xy(group)
                if xy is None:
                    continue
                if want_motion:
                    speed = parse_speed(group)
                    if speed is not None:
                        speeds.append(speed)
                team = int(slot_teams[slot])
                slot_counts[frame, team] += 1
                outside_player += int(any(v < 0 or v > 1 for v in xy))
                if ball_m is None:
                    continue
                delta = np.asarray(xy) * scale - ball_m
                distance = float(np.sqrt(np.dot(delta, delta)))
                if distance < nearest_dist[frame, team]:
                    nearest_dist[frame, team] = distance
                    nearest_slot[frame, team] = slot
            if want_motion and speeds:
                speed_median[frame] = np.median(speeds)
                speed_p90[frame] = np.percentile(speeds, 90)
                speed_count[frame] = len(speeds)
            count += 1
    if count != n_frames:
        raise ValueError("Tracking record count does not match the tactical-video length.")
    ball_visible = np.all(np.isfinite(ball_xy), axis=1)
    geometry_available = np.isfinite(nearest_dist)
    features = {
        "frame": np.arange(n_frames, dtype=np.int32),
        "ball_xy_m": ball_xy,
        "ball_visible": ball_visible,
        "nearest_dist_m": nearest_dist,
        "nearest_slot": nearest_slot,
        "team_visible_slot_count": slot_counts,
        "geometry_available": geometry_available,
        "slot_team": slot_teams,
        "pitch_m": scale,
    }
    summary = {
        "n_frames": n_frames,
        "tracking_ids_exact": True,
        "n_track_slots": int(len(slot_teams)),
        "ball_visible_frames": int(ball_visible.sum()),
        "ball_and_any_player_frames": int(geometry_available.any(axis=1).sum()),
        "both_teams_geometry_frames": int(geometry_available.all(axis=1).sum()),
        "finite_ball_observations_outside_unit_square": outside_ball,
        "finite_player_observations_outside_unit_square": outside_player,
    }
    if want_motion:
        motion_out.update({
            "frame": np.arange(n_frames, dtype=np.int32),
            "player_speed_median_mps": speed_median,
            "player_speed_p90_mps": speed_p90,
            "player_speed_count": speed_count,
        })
    return features, summary


def audit_tags(events, n_frames):
    for event in events:
        if not (np.isfinite(event["start_s"]) and np.isfinite(event["end_s"])):
            raise ValueError("A tagging interval has nonfinite timestamps.")
        if event["start_frame"] >= event["end_frame"]:
            raise ValueError("A tagging interval is empty or reversed.")
    # Legacy helper uses inclusive-end containment; preserve its report only.
    report = check_time_base(events)
    # The right-exclusive loop below is the sole kickoff-clock acceptance gate.
    for half, kickoff in KICKOFF_VIDEO_FRAMES.items():
        matched = any(
            (event["code"] or "").strip().casefold() == "set pieces"
            and (event["tags"].get("Type") or "").strip().casefold() == "kick off (start)"
            and event["tags"].get("Half") == half
            and event["start_frame"] <= kickoff < event["end_frame"]
            for event in events
        )
        if not matched:
            raise ValueError("Kickoff containment failed under right-exclusive intervals.")
    return {
        "n_events": len(events),
        "clock_check": report,
        "right_exclusive_kickoff_check": True,
        "intervals_outside_video": sum(
            event["start_frame"] < 0 or event["end_frame"] > n_frames
            for event in events
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME, choices=sorted(GAMES))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "local_cache"))
    args = parser.parse_args()
    n_frames = source_frames("tactical", args.game)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], args.game, args.data_dir)
    names, slot_teams = read_metadata(meta_path)
    print("Reading tracking and validating every frame ID...")
    motion = {}
    features, tracking_report = extract_tracking(
        raw_path, slot_teams, n_frames, GAMES[args.game]["pitch"], motion_out=motion
    )
    print("PASS: tracking IDs, record count, slot schema, and team assignments.")
    events = load_tags(game=args.game, data_dir=args.data_dir)
    tagging_report = audit_tags(events, n_frames)
    print("PASS: tagging intervals and published kickoff clock checks.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "tracking_features.npz", **features)
    # Separate file, so tracking_features.npz (and everything built on it) is unchanged.
    np.savez_compressed(out_dir / "player_motion.npz", schema_version=np.int32(1), **motion)
    report = {
        "schema_version": 1,
        "game": args.game,
        "team_order": {"column_0": "H", "column_1": "A"},
        "team_names": names,
        "tracking": tracking_report,
        "tagging": tagging_report,
        "notes": [
            "Track slots are not stable player identities or unique player counts.",
            "Missing geometry is NaN for ball coordinates and inf for nearest distance.",
            "geometry_available must mask distances before evidence calculation.",
            "Outside-pitch coordinates are recorded, not automatically discarded.",
            "No ground-truth CSV was read; no possession labels were generated.",
            "Generated files are restricted-data derivatives and must stay private.",
        ],
    }
    with open(out_dir / "stage2_audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Saved local tracking_features.npz, player_motion.npz and stage2_audit.json.")
    print("Keep the output directory Git-ignored. No new possession CSV was produced.")


if __name__ == "__main__":
    main()
