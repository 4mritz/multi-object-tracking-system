import numpy as np
from .iou import compute_iou

def build_cost_matrix(tracks, detections):
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = compute_iou(track.bbox, det)
            cost_matrix[i, j] = 1 - iou  # lower is better

    return cost_matrix