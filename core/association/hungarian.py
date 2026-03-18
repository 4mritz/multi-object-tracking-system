from scipy.optimize import linear_sum_assignment

def match(cost_matrix, iou_threshold=0.3):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > (1 - iou_threshold):
            unmatched_tracks.append(r)
            unmatched_detections.append(c)
        else:
            matches.append((r, c))

    return matches, unmatched_tracks, unmatched_detections