"""
This module returns transformed frames from the webcam

Frames are warped in perspective and reshaped to be usable in prediction
"""
import config
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
from imutils import perspective


class Transformer:
    def __init__(self, capture_channel: int):
        """
        Initializes webcam capture

        Args:
            capture_channel: source of webcam feed
        """
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()

        self.cap = cv.VideoCapture(capture_channel)

    def get_transformed_image(self):
        """
        Performs warp perspective transformation on webcam frame

        Returns:
            warped and reshaped frame (64 x 64) for prediction

            None if less than 4 AruCo markers are detected
        """
        ret, frame = self.cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) == 4:
            points = []
            for marker in corners:
                points.append(marker[0][2].tolist())
            points = perspective.order_points(np.array(points))
            destination = np.float32(
                np.array([[0, 0], [self.WINDOW_WIDTH, 0], [self.WINDOW_WIDTH, self.WINDOW_HEIGHT], [0, self.WINDOW_HEIGHT]]))
            matrix = cv.getPerspectiveTransform(points, destination)
            frame = cv.warpPerspective(
                frame, matrix, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), flags=cv.INTER_LINEAR)

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            resized = cv.resize(rgb_frame, config.SIZE)
            reshaped = resized.reshape(-1, config.IMG_SIZE,
                                       config.IMG_SIZE, config.COLOR_CHANNELS)

            return reshaped
        else:
            return None
