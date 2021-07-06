import numpy as np
from pyntcloud import PyntCloud
import pandas as pd

# Load PC and get coordinates
def load_pc(name):
    in_pc = PyntCloud.from_file(name)
    points = in_pc.xyz
    points = points.astype(np.uint32)

    return points


# Write PC to PLY file
def save_pc(pts_geom, name):
    pts_geom = pts_geom.astype(np.float32)
    geom = {'x': pts_geom[:, 0], 'y': pts_geom[:, 1], 'z': pts_geom[:, 2]}
    cloud = PyntCloud(pd.DataFrame(data=geom))
    cloud.to_file(name)


# Divide PC into blocks
def pc2blocks(points, blk_size):
    blk_map_full = points // blk_size
    blk_map, point2blk_idx, blk_len = np.unique(blk_map_full, return_inverse=True, return_counts=True, axis=0)
    num_blocks = blk_map.shape[0]
    blk_p_coord = points % blk_size

    blocks = [np.zeros((i, 3)) for i in blk_len]
    for i in range(num_blocks):
        blocks[i] = blk_p_coord[point2blk_idx == i, :]

    return blocks, blk_map


# Convert point coordinates into a block of binary voxels
def point2vox(block, blk_size):
    blk_size = np.int32(blk_size)
    block = block.astype(np.int32)
    output = np.zeros([1, blk_size, blk_size, blk_size, 1], dtype=np.float32)
    output[0, block[:, 0], block[:, 1], block[:, 2], 0] = 1.0

    return output


# Convert block of binary voxels into point coordinates
def vox2point(block):
    idx_a, idx_b, idx_c = np.where(block)
    points = np.stack((idx_a, idx_b, idx_c), axis=1)

    return points
