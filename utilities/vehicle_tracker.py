import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag


class VehicleTracker:
    # Method: Constructor
    def __init__(self):
        # Initialize parameters for tracker
        self.id = 0
        self.num_hits = 0
        self.num_unmatched = 0
        self.box = []

        # Initialize parameters for the Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=8)
        self.dt = 1.0
        self.x_state = []

        # State transition matrix (assuming constant velocity model)
        self.kf.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, self.dt, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, self.dt, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, self.dt],
                              [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix (assuming we can only measure the coordinates)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]])

        # State covariance matrix
        self.kf.P *= 100.0

        # Process uncertainty
        self.Q_comp_mat = np.array([[self.dt ** 4 / 2., self.dt ** 3 / 2.],
                                    [self.dt ** 3 / 2., self.dt ** 2]])
        self.kf.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                               self.Q_comp_mat, self.Q_comp_mat)

        # State uncertainty
        self.kf.R = np.eye(4)*6.25

    # Method: Used to predict and update the next state for a bounding box
    def predict_and_update(self, z):
        """
        :param z: Box
        :return: Box with updated location
        """
        self.kf.x = self.x_state

        # Predict
        self.kf.predict()

        # Update
        self.kf.update(z)

        # Get current state and convert to integers for pixel coordinates
        self.x_state = self.kf.x.astype(int)

    # Method: Used to only predict the next state for the bounding box
    def predict(self):
        """
        :return: Box with predicted location
        """
        self.kf.x = self.x_state

        # Predict
        self.kf.predict()

        # Get current state and convert to integers for pixel coordinates
        self.x_state = self.kf.x.astype(int)
