import numpy as np
from core.kalman.kalman_filter import KalmanFilter
from core.tracking.track import Track
from core.association.cost_matrix import build_cost_matrix
from core.association.hungarian import match
from core.utils.bbox import xyxy_to_cxcywh

class MultiObjectTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.kf = KalmanFilter()
        self.tracks = []
        self.next_id = 0

        self.max_age = max_age
        self.min_hits = min_hits

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        measurements = [xyxy_to_cxcywh(det) for det in detections]

        if len(self.tracks) == 0:
            for m in measurements:
                x, P = self.kf.initiate(m)
                self.tracks.append(Track(self.next_id, x, P))
                self.next_id += 1
            return

        cost_matrix = build_cost_matrix(self.tracks, detections)
        matches, unmatched_tracks, unmatched_detections = match(cost_matrix)

        # Update matched
        for t, d in matches:
            self.tracks[t].update(self.kf, measurements[d])

        # Create new tracks
        for d in unmatched_detections:
            x, P = self.kf.initiate(measurements[d])
            self.tracks.append(Track(self.next_id, x, P))
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks if t.time_since_update < self.max_age
        ]

    def get_active_tracks(self):
        return [
            t for t in self.tracks
            if t.hits >= self.min_hits and t.time_since_update == 0
        ]