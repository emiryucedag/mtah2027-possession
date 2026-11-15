"""Reader for the tactical Automatic Tracking Data (Metrica FIFA/EPTS export, metadata XML + raw text).

A raw line is `frame_id : 46 player slots separated by ';' : ball`, every group `x,y,speed` with NaN for a missing
observation, coordinates normalised to 0-1. `frame_id` counts from 1, a submission row from 0.
The slots are tracklets, not people: they appear, vanish and swap identity, so only their team is meaningful.
"""

import xml.etree.ElementTree as ET

import numpy as np


def slot_teams(meta_path):
    """Array of 'H' / 'A' per slot. The first team in the metadata is the home side (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array(["H" if p.get("teamId") == home else "A" for p in root.iter("Player")], dtype=object)


def read_atd(raw_path, n_frames):
    """(players (n, slots, 2), ball (n, 2)) in normalised coordinates, NaN where unobserved."""
    players = ball = None
    with open(raw_path) as f:
        for line in f:
            frame, slots, b = line.rstrip("\n").split(":")
            i = int(frame) - 1
            if players is None:
                n_slots = slots.count(";") + 1
                players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
                ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
            for s, group in enumerate(slots.split(";")):
                if group[0] != "N":
                    x, y, _ = group.split(",")
                    players[i, s] = float(x), float(y)
            if b[0] != "N":
                x, y, _ = b.split(",")
                ball[i] = float(x), float(y)
    return players, ball


def home_attack_sign(meta_path):
    """+1 if the home team attacks towards +x in the first half (left_to_right), else -1."""
    root = ET.parse(meta_path).getroot()
    home = next(iter(root.iter("Team")))
    for param in home.iter("ProviderParameter"):
        if param.findtext("Name") == "attack_direction_first_half":
            return 1.0 if param.findtext("Value") == "left_to_right" else -1.0
    raise ValueError("attack_direction_first_half not found in " + meta_path)
