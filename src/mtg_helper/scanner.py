"""
This class is responsible for capturing the raw images that will be
processed by the image recognition pipeline.
"""

import time

import cv2
from cv2.typing import MatLike


class CameraError(Exception):
    """base camera exception"""


class CameraNotOpenedError(CameraError):
    """Raised during camera setup if selected device is not open"""


class ImageNotCapturedError(CameraError):
    """Raised when an image is not succesffuly captured during a burst"""


class Scanner:
    def __init__(self, camera_input: int = 0) -> None:
        self._burst_count: int = 10
        self._camera_input: int = camera_input
        self.camera: cv2.VideoCapture = cv2.VideoCapture(camera_input)
        self._setup_camera()

    def _setup_camera(self) -> None:
        """Open the camera device and discard any stale frames sitting in its buffer.

        See docs/image_pipeline_reference.md for implementation details.
        """
        # make sure camera is open, flush frame buffer
        if not self.camera.isOpened():
            raise CameraNotOpenedError(f"camera device {self._camera_input} not opened")
        for _ in range(5):
            self.camera.grab()

    def capture_burst(self) -> list[MatLike]:
        """Warm up the camera's autofocus, then capture a short burst of frames.

        See docs/image_pipeline_reference.md for implementation details.
        """
        # allow autofocus to warm up
        for _ in range(15):
            self.camera.read()
        time.sleep(0.5)

        # capture image burst
        new_burst: list[MatLike] = []
        for _ in range(self._burst_count):
            is_captured, image = self.camera.read()
            if not is_captured:
                raise ImageNotCapturedError("burst image capture failed")
            new_burst.append(image)
        return new_burst
