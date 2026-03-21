import numpy as np
from core.kalman.kalman_filter import KalmanFilter
from core.tracking.track import Track
from core.association.cost_matrix import build_cost_matrix
from core.association.hungarian import match
from core.utils.bbox import xyxy_to_cxcywh
from models.appearance.encoder import AppearanceEncoder

class MultiObjectTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.kf = KalmanFilter()
        self.tracks = []
        self.next_id = 0

        self.encoder = AppearanceEncoder()  # NEW

        self.max_age = max_age
        self.min_hits = min_hits

    def predict(self):   
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, frame, detections):
        measurements = [xyxy_to_cxcywh(det) for det in detections]

        det_features = self.encoder.extract(frame, detections)

        if len(self.tracks) == 0:
            for m, f in zip(measurements, det_features):
                x, P = self.kf.initiate(m)
                self.tracks.append(Track(self.next_id, x, P, f))
                self.next_id += 1
            return

        cost_matrix = build_cost_matrix(self.tracks, detections, det_features)

        matches, unmatched_tracks, unmatched_detections = match(cost_matrix)

        for t, d in matches:
            self.tracks[t].update(self.kf, measurements[d], det_features[d])

        for d in unmatched_detections:
            x, P = self.kf.initiate(measurements[d])
            self.tracks.append(Track(self.next_id, x, P, det_features[d]))
            self.next_id += 1

        self.tracks = [
            t for t in self.tracks if t.time_since_update < self.max_age
        ]