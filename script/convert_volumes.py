"""Convert MMHPSD raw event .npz files into 4-channel event volume images."""
import argparse
import os
import cv2
import torch
import numpy as np
import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_utils.event_utils import gen_discretized_event_volume_no_polarity, normalize_event_volume


def convert_volumes(args):
    seq, event_file, event_folder, save_folder = args
    events = np.load(os.path.join(event_folder, seq, "events", event_file))
    events = np.hstack([events['xy'], events['t'].reshape(-1, 1), events['p'].reshape(-1, 1)])
    volume = gen_discretized_event_volume_no_polarity(torch.tensor(events), (4, 256, 256))
    volume = normalize_event_volume(volume)
    volume = volume.permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(save_folder, seq, event_file.replace(".npz", ".png")), volume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_folder", type=str, default="data/mmhpsd_events")
    parser.add_argument("--save_folder", type=str, default="data/mmhpsd_volumes")
    args = parser.parse_args()

    event_folder = args.event_folder
    save_folder = args.save_folder

    seqs = [seq for seq in os.listdir(event_folder) if seq[0] != "."]
    jobs = []

    for seq in seqs:
        events_dir = os.path.join(event_folder, seq, "events")
        if not os.path.isdir(events_dir):
            continue
        events = [f for f in os.listdir(events_dir) if f[0] != "."]

        save_seq_folder = os.path.join(save_folder, seq)
        if not os.path.exists(save_seq_folder):
            os.makedirs(save_seq_folder)

        for event_file in events:
            jobs.append((seq, event_file, event_folder, save_folder))

    for job in tqdm.tqdm(jobs):
        convert_volumes(job)
