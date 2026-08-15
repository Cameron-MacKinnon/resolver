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
    """Raised when an image is not successfully captured during a burst"""


class Scanner:
    def __init__(self, camera_input: int = 0) -> None:
        self._burst_count: int = 10
        self._camera_input: int = camera_input
        self.camera: cv2.VideoCapture = cv2.VideoCapture(camera_input)
        self._setup_camera()

    def __enter__(self) -> "Scanner":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def release(self) -> None:
        """Release the underlying camera device so other processes/apps can use it."""
        self.camera.release()

    def _setup_camera(self) -> None:
        """Open the camera device and discard any stale frames sitting in its buffer.

        `grab()` reads a frame into the device's internal buffer without
        decoding it (cheaper than `read()`, which grabs and decodes) - used
        here purely to flush frames that were sitting in the buffer before
        we were ready, or captured under stale auto-exposure settings.
        """
        # make sure camera is open, flush frame buffer
        if not self.camera.isOpened():
            raise CameraNotOpenedError(f"camera device {self._camera_input} not opened")
        for _ in range(5):
            self.camera.grab()

    def capture_burst(self) -> list[MatLike]:
        """Warm up the camera's autofocus, then capture a short burst of frames.

        The first frames after opening a device are often still soft or
        mis-exposed while autofocus/auto-exposure settle, so several `read()`
        calls are discarded as a warm-up before the real capture. A burst of
        frames is captured (rather than one) so FrameSelector has several
        candidates to pick the sharpest from, since any single frame -
        including the last warm-up read - might still catch motion blur or
        a focus hunt.
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
