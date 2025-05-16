#!/usr/bin/env python3
import math
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
#import open_clip

from lightglue import SuperPoint          
import cv2       

import sys
from PIL import Image

# Count-Min Sketch
class InformationSketch:
    # Tiny (sum depth * width) struct, counts token freqs, computes an entropy metric (how much have we seen so far)
    def __init__(self, width: int = 16_384, depth: int = 4, seed: int = 0):
        self.W, self.D = width, depth
        self.table = np.zeros((depth, width), dtype=np.uint32)
        rng = random.Random(seed)
        self.salts = [rng.getrandbits(32) for _ in range(depth)] # k 32 bit hash salts
        self.total = 0  # tokens seen thus far

    def _hash(self, token: int, d: int) -> int:
        # simple hash: mult, shift, mod
        return ((token ^ self.salts[d]) * 0x9e3779b1) & 0xFFFFFFFF % self.W

    def update(self, tokens: List[int]) -> None:
        # bag of tokens, with duplicates OK
        for t in tokens:
            for d in range(self.D):
                self.table[d, self._hash(t, d)] += 1
        self.total += len(tokens)

    # compute information density
    def entropy(self) -> float:
        # Shannon entropy of sketch counts
        if self.total == 0:
            return 0.0
        mins = self.table.min(axis=0).astype(np.float64)
        probs = mins / self.total
        nonzero = probs > 0
        return -np.sum(probs[nonzero] * np.log2(probs[nonzero]))                         

# we implemented a version of the matching mechanism using clip, but as this project's first aim is integration into GaussNav, we currently use lightglue
def build_lightglue_extractor(device: str = "cuda:0"):
    # Returns a list of 64 bit ints (hashes for each feature descriptor)
    model = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # random projection
    rng = torch.Generator(device).manual_seed(42)
    build_lightglue_extractor.proj = torch.empty((256, 64), device=device) \
                                         .uniform_(-1.0, 1.0).sign_()

    def _bits_to_int(bits):
        # row --> single value
        int_values = []
        # bits = bits.cpu()
        # print(bits)
        # print(len(bits))
        for bit_row in bits[0]:
            value = 0
            for i in range(64):
                # print(bit_row[i])
                if bit_row[i].item():
                    value |= (1 << (i % 64))
            int_values.append(value)
        return int_values


    def extract(rgb: np.ndarray) -> List[int]:
        # rgb to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        im_t = torch.from_numpy(gray)[None, None].float().to(device) / 255.0

        with torch.no_grad():
            feats = model.extract(im_t)              # descriptors
            if feats["descriptors"].numel() == 0:    # if none, return empty list
                return []
            desc = F.normalize(feats["descriptors"], dim=-1)     

            # return lsh
            sign_bits = (desc @ build_lightglue_extractor.proj > 0)
        return _bits_to_int(sign_bits)

    return extract


# example usage
if __name__ == "__main__":
    # load .npz from collect_trajectories.py, and print entropy growth for a single trajectory

    if len(sys.argv) != 2:
        print("Please pass the path, like so: python information_computation.py trajectories/habitat_views.npz")
        sys.exit(1)

    archive = np.load(sys.argv[1], allow_pickle=True)
    traj = archive['circle_in']          

    # sample just 30 frames (without replacement)t)
    if len(traj) < 30:
        raise ValueError(f"trajectory only has {len(traj)} frames, need at least 30 to sample.")
    idx = np.random.choice(len(traj), size=30, replace=False)
    idx.sort()                      
    views = [traj[i]['rgb'] for i in idx]
    #views = [obs['rgb'] for obs in traj]

    extractor = build_lightglue_extractor()
    sketch = InformationSketch()

    for i, rgb in enumerate(views, 1):
        sketch.update(extractor(rgb))
        print(f"frame {i:03d}, total entropy of path = {sketch.entropy():.3f}")