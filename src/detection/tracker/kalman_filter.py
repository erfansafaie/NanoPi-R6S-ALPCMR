
import numpy as np
import scipy.linalg

class KalmanFilterXYWH():
    """A KalmanFilterXYWH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a Kalman filter for tracking bounding boxes with state space (x, y, w, h, vx, vy, vw, vh), where (x, y)
    is the center position, w is the width, h is the height, and vx, vy, vw, vh are their respective velocities. The
    object motion follows a constant velocity model, and the bounding box location (x, y, w, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Create a track from an unassociated measurement.
        predict: Run the Kalman filter prediction step.
        project: Project the state distribution to measurement space.
        multi_predict: Run the Kalman filter prediction step in a vectorized manner.
        update: Run the Kalman filter correction step.

    Examples:
        Create a Kalman filter and initialize a track
        >>> kf = KalmanFilterXYWH()
        >>> measurement = np.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        The Kalman filter is initialized with an 8-dimensional state space (x, y, a, h, vx, vy, vw, vh), where (x, y)
        represents the bounding box center position, 'a' is the aspect ratio, 'h' is the height, and their respective
        velocities are (vx, vy, vw, vh). The filter uses a constant velocity model for object motion and a linear
        observation model for bounding box location.
        """
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and
                height.

        Returns:
            mean (np.ndarray): Mean vector (8 dimensional) of the new track. Unobserved velocities are initialized to 0
                mean.
            covariance (np.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> measurement = np.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
            >>> print(mean)
            [100.  50.  20.  40.   0.   0.   0.   0.]
            >>> print(covariance)
            [[ 4.  0.  0.  0.  0.  0.  0.  0.]
             [ 0.  4.  0.  0.  0.  0.  0.  0.]
             [ 0.  0.  4.  0.  0.  0.  0.  0.]
             [ 0.  0.  0.  4.  0.  0.  0.  0.]
             [ 0.  0.  0.  0.  0.25  0.  0.  0.]
             [ 0.  0.  0.  0.  0.  0.25  0.  0.]
             [ 0.  0.  0.  0.  0.  0.  0.25  0.]
             [ 0.  0.  0.  0.  0.  0.  0.  0.25]]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (np.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time
                step.

        Returns:
            mean (np.ndarray): Mean vector of the predicted state. Unobserved velocities are initialized to 0 mean.
            covariance (np.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            mean (np.ndarray): Projected mean of the given state estimate.
            covariance (np.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (np.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            mean (np.ndarray): Mean matrix of the predicted states with shape (N, 8).
            covariance (np.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> mean = np.random.rand(5, 8)  # 5 objects with 8-dimensional state vectors
            >>> covariance = np.random.rand(5, 8, 8)  # 5 objects with 8x8 covariance matrices
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            new_mean (np.ndarray): Measurement-corrected state mean.
            new_covariance (np.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
