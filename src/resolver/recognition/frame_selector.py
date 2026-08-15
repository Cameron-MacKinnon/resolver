import cv2
from cv2.typing import MatLike


class FrameSelectionError(Exception):
    """Base frame selection exception"""


class NoSharpestImageError(FrameSelectionError):
    """Raised when no 'sharpest image' was found in the batch (this should not occur)"""


class FrameSelector:
    def __init__(self, images: list[MatLike]) -> None:
        self.batch: list[MatLike] = images
        self.highest_score: float = -1
        self.sharpest_image: MatLike | None = None

    def _convert_image_to_grayscale(self, image: MatLike) -> MatLike:
        """Convert an image to grayscale if not already (necessary for Laplacian variance)."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image

    def select_sharpest_image(self) -> MatLike:
        """Pick the least-blurry frame out of self.batch using Laplacian variance.

        Laplacian variance is a standard blur-detection heuristic: the
        Laplacian highlights edges, and a sharp image has many strong,
        varied edges while a blurry one has fewer/weaker ones, so the
        variance of the Laplacian is highest for the sharpest frame.
        """
        for index, image in enumerate(self.batch):
            grayscale_image = self._convert_image_to_grayscale(image)

            # compute sharpness score (Laplacian variance)
            score: float = float(cv2.Laplacian(grayscale_image, cv2.CV_64F).var())

            # update highest score
            if score > self.highest_score:
                self.highest_score = score
                self.sharpest_image = image

        # raise an error if something has gone horribly wrong here and there
        # is somehow no sharpest image
        if self.sharpest_image is None:
            raise NoSharpestImageError

        return self.sharpest_image
