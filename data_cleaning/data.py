import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

PAD = 0
MAX_LEN = 70

class RallyDataset(Dataset):

    def __init__(self, df):
        group = df.groupby('rally_id').apply(
            lambda r: (r['type'].values, r['landing_x'].values, 
                       r['landing_y'].values, r['player'].values)
        )
        self.rally_ids = list(group.index)
        self.sequences = dict(group)

    def __len__(self):
        return len(self.rally_ids)

    # returns 
    def __getitem__(self, index):
        rally_id = self.rally_ids[index]
        shot_type, landing_x, landing_y, player = self.sequences[rally_id]
        rally_len = len(shot_type) - 1  # -1 because input/target shift

        # preallocate padded arrays
        inp_shot = np.zeros(MAX_LEN, dtype=int)
        inp_x    = np.zeros(MAX_LEN, dtype=float)
        inp_y    = np.zeros(MAX_LEN, dtype=float)
        inp_player = np.zeros(MAX_LEN, dtype=int)
        tgt_shot = np.zeros(MAX_LEN, dtype=int)
        tgt_x    = np.zeros(MAX_LEN, dtype=float)
        tgt_y    = np.zeros(MAX_LEN, dtype=float)
        tgt_player = np.zeros(MAX_LEN, dtype=int)

        # fill real data (shifted by 1)
        n = min(rally_len, MAX_LEN)
        inp_shot[:n]   = shot_type[:-1][:n]
        inp_x[:n]      = landing_x[:-1][:n]
        inp_y[:n]      = landing_y[:-1][:n]
        inp_player[:n] = player[:-1][:n]
        tgt_shot[:n]   = shot_type[1:][:n]
        tgt_x[:n]      = landing_x[1:][:n]
        tgt_y[:n]      = landing_y[1:][:n]
        tgt_player[:n] = player[1:][:n]

        return (inp_shot, inp_x, inp_y, inp_player,
                tgt_shot, tgt_x, tgt_y, tgt_player, n)