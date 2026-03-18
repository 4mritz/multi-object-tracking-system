from core.utils.bbox import cxcywh_to_xyxy

class Track:
    def __init__(self, track_id, x, P):
        self.id = track_id
        self.x = x
        self.P = P

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def predict(self, kf):
        self.x, self.P = kf.predict(self.x, self.P)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, measurement):
        self.x, self.P = kf.update(self.x, self.P, measurement)
        self.hits += 1
        self.time_since_update = 0

    @property
    def bbox(self):
        return cxcywh_to_xyxy(self.x[:4].flatten())