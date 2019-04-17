import numpy as np
import torch
import torch.nn as nn

def decode_segmap(label_mask):
	r = label_mask.copy()
	g = label_mask.copy()
	b = label_mask.copy()
	rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
	rgb[:, :, 0] = r * 255.0
	rgb[:, :, 1] = g * 255.0
	rgb[:, :, 2] = b * 255.0
	rgb = np.rint(rgb).astype(np.uint8)
	return rgb

def decode_seg_map_sequence(label_masks):
	assert(label_masks.ndim == 3 or label_masks.ndim == 4)
	if label_masks.ndim == 4:
		label_masks = label_masks.squeeze(1)
	assert(label_masks.ndim == 3)
	rgb_masks = []
	for i in range(label_masks.shape[0]):
		rgb_mask = decode_segmap(label_masks[i])
		rgb_masks.append(rgb_mask)
	rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
	return rgb_masks
