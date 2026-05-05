import numpy as np


class KalmanFilterXYWH:
    """
    8D state:
        x = [cx, cy, w, h, vx, vy, vw, vh]

    4D measurement:
        z = [cx, cy, w, h]
    """

    def __init__(self):
        ndim = 4
        dt = 1.0

        self._motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        """
        measurement: [cx, cy, w, h]
        """
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement

        std = np.array([
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ], dtype=np.float32)

        covariance = np.diag(std * std).astype(np.float32)
        return mean, covariance

    def predict(self, mean, covariance):
        # Prevent degenerate width/height
        mean = mean.copy()
        mean[2] = max(1.0, mean[2])
        mean[3] = max(1.0, mean[3])

        std_pos = np.array([
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ], dtype=np.float32)

        std_vel = np.array([
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ], dtype=np.float32)

        motion_cov = np.diag(np.concatenate([std_pos, std_vel]) ** 2).astype(np.float32)

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        mean[2] = max(1.0, mean[2])
        mean[3] = max(1.0, mean[3])

        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean, covariance):
        mean = mean.copy()
        mean[2] = max(1.0, mean[2])
        mean[3] = max(1.0, mean[3])

        std = np.array([
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ], dtype=np.float32)

        innovation_cov = np.diag(std * std).astype(np.float32)

        mean_proj = self._update_mat @ mean
        cov_proj = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        return mean_proj.astype(np.float32), cov_proj.astype(np.float32)

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        # More numerically stable than direct full inverse usage pattern in some cases
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ self._update_mat @ covariance

        new_mean[2] = max(1.0, new_mean[2])
        new_mean[3] = max(1.0, new_mean[3])

        return new_mean.astype(np.float32), new_covariance.astype(np.float32)
