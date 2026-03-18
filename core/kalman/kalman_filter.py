import numpy as np

class KalmanFilter:
    def __init__(self):
        dt = 1.0

        # State: [cx, cy, w, h, vx, vy]
        self.F = np.eye(6)
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        self.P = np.eye(6) * 10
        self.Q = np.eye(6)
        self.R = np.eye(4)

    def initiate(self, measurement):
        x = np.zeros((6, 1))
        x[:4, 0] = measurement
        return x, self.P.copy()

    def predict(self, x, P):
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x, P, z):
        z = z.reshape(4, 1)
        y = z - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(len(P)) - K @ self.H) @ P
        return x, P