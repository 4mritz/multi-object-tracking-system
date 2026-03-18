from core.utils.bbox import cxcywh_to_xyxy
import numpy as np

class Track:
    def __init__(self, track_id, x, P, feature=None):
        self.id = track_id
        self.x = x
        self.P = P

        self.feature = feature  # NEW

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def update(self, kf, measurement, feature=None):
        self.x, self.P = kf.update(self.x, self.P, measurement)

        if feature is not None:
            if self.feature is None:
                self.feature = feature
            else:
                self.feature = 0.8 * self.feature + 0.2 * feature
                self.feature /= np.linalg.norm(self.feature) + 1e-6

        self.hits += 1
        self.time_since_update = 0