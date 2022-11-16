# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import os
import sys
sys.path.append(os.getcwd())

import os
import os.path as osp
import joblib
import numpy as np
from tqdm import tqdm

# extract 24 SMPL joints from 52 SMPL-H joints
joints_to_use = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 37]
)
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

all_sequences = [
    "ACCAD",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    "BMLhandball",
    "DanceDB"
]

def read_data(folder, sequences):

    if sequences == "all":
        sequences = all_sequences

    db = {}
    for seq_name in sequences:
        print(f"Reading {seq_name} sequence...")
        seq_folder = osp.join(folder, seq_name)

        datas = read_single_sequence(seq_folder, seq_name)
        db.update(datas)
        print(seq_name, "number of seqs", len(datas))

    return db

def read_single_sequence(folder, seq_name):
    subjects = os.listdir(folder)

    datas = {}

    for subject in tqdm(subjects):
        actions = [
            x for x in os.listdir(osp.join(folder, subject)) if x.endswith(".npz") and osp.isdir(osp.join(folder, subject))
        ]

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith("shape.npz"):
                continue

            data = dict(np.load(fname))

            new_data = {}
            new_data['poses'] = data['poses'][:, joints_to_use]
            new_data['trans'] = data['trans']
            new_data['mocap_framerate'] = data['mocap_framerate']

            vid_name = f"{seq_name}_{subject}_{action[:-4]}"

            datas[vid_name] = new_data

    return datas


if __name__ == "__main__":

    amass_dir = "E:/PanLiang_Datasets/AMASS"
    out_dir = "./data/motion"
    sequences = [
        "ACCAD"
    ]

    db = read_data(amass_dir, sequences)
    db_file = osp.join(out_dir, "amass_db.pkl")
    print(f"Saving AMASS dataset to {db_file}")
    joblib.dump(db, db_file)
