import numpy as np
from .iou import compute_iou

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def build_cost_matrix(tracks, detections, det_features=None,
                      w_iou=0.5, w_app=0.4, w_motion=0.1):

    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):

            iou = compute_iou(track.bbox, det)
            iou_cost = 1 - iou

            app_cost = 0
            if det_features is not None and track.feature is not None:
                app_cost = cosine_distance(track.feature, det_features[j])

            # simple motion cost (distance between centers)
            tx, ty = track.x[0,0], track.x[1,0]
            dx = (det[0] + det[2]) / 2
            dy = (det[1] + det[3]) / 2
            motion_cost = np.linalg.norm([tx - dx, ty - dy]) / 1000

            cost = (
                w_iou * iou_cost +
                w_app * app_cost +
                w_motion * motion_cost
            )

            cost_matrix[i, j] = cost

    return cost_matrix