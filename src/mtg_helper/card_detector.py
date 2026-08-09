from collections.abc import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

from .card_geometry import CARD_HEIGHT_PX, CARD_RATIO, CARD_WIDTH_PX


class CardDetector:
    def __init__(self, image: MatLike) -> None:
        self.image: MatLike = image
        self.corners: np.ndarray | None = None

    def _resize_image(self) -> MatLike:
        # set the desired length of longer edge for resize
        target_width = 1000

        # get the image's original dimensions
        original_height, original_width = self.image.shape[:2]

        # calculate aspect ratio and use it to scale target height
        aspect_ratio = original_height / original_width
        target_height = int(target_width * aspect_ratio)

        # resize image and return
        resized_image = cv2.resize(
            self.image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
        return resized_image

    def _preprocess_image(self, resized_image: MatLike) -> MatLike:
        # convert image to from BGR to LAB colour space and to seprate brightness
        # and colour information (we only care about normalising brightness here)
        lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Create CLAHE object and apply to lightness channel specifically,
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_l_channel = clahe.apply(l_channel)

        # merge the channels back together now that CLAHE has been applied to
        # lightness channel
        lab_image_processed = cv2.merge((clahe_l_channel, a_channel, b_channel))

        # convert back to BGR colour space and return
        return cv2.cvtColor(lab_image_processed, cv2.COLOR_LAB2BGR)

    def _find_contours(self, preprocessed_image: MatLike) -> Sequence[MatLike]:
        # convert image to grayscale
        gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

        # apply light gaussian blur to smooth out print dots
        blurred_gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # apply adaptive gaussian thresholding
        thresholded_image = cv2.adaptiveThreshold(
            src=blurred_gray_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=51,
            C=3,
        )

        # find and return all contours (discard hierarchy, we're not using it here)
        contours, _ = cv2.findContours(
            thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _filter_contours_to_cardlike_quadrilaterals(
        self, contours: Sequence[MatLike]
    ) -> list[np.ndarray]:
        noise_cutoff = 500  # contours with areas smaller than this will be ignored
        solidity_cutoff = 0.95  # shapes with solidity below this cutoff will be ignored
        ratio_drift = 0.15  # allowed difference in aspect ratio from expectations

        # final list of quadrilateral corners that match our definition of a card
        cardlike_quads = []

        for c in contours:
            # get the area of the current contour
            area = cv2.contourArea(c)

            # filter out tiny noise areas
            if area < noise_cutoff:
                continue

            # compute convex hull and solidity
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area

            # mtg cards are highly solid rectangles (approaching 1.0), filter for this
            if solidity < solidity_cutoff:
                continue

            # at this point we can assume it's a rectangle, get its bounding box
            # (accounting for rotation)
            rect = cv2.minAreaRect(hull)
            (_, _), (width, height), _ = rect

            # prevent accidental ZeroDivisionErrors in the following block
            if width == 0 or height == 0:
                continue

            # calcuate orientation-agnostic aspect ratio
            aspect_ratio = min(width, height) / max(width, height)
            if (CARD_RATIO - ratio_drift) <= aspect_ratio <= (CARD_RATIO + ratio_drift):
                # approximate the hull's own corners instead of forcing a rectangle -
                # a card viewed at an angle projects as a general quadrilateral, not a
                # true rectangle, so minAreaRect's boxPoints would distort that skew
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

                # only accept it if simplification actually yielded 4 corners
                if len(approx) == 4:
                    cardlike_quads.append(approx.reshape(4, 2).astype("float32"))

        # resulting list should only contain quads that closely resemble a card
        return cardlike_quads

    def _order_points(self, quad: np.ndarray) -> np.ndarray:
        """Sort a quad's 4 corners into [top-left, top-right, bottom-right, bottom-left] order."""
        # start with an empty slot for each of the 4 ordered corners
        rectangle = np.zeros((4, 2), dtype="float32")

        # top-left has the smallest x+y, bottom-right has the largest
        s = quad.sum(axis=1)
        rectangle[0] = quad[np.argmin(s)]  # top-left
        rectangle[2] = quad[np.argmax(s)]  # bottom-right

        # top-right has the smallest x-y, bottom-left has the largest
        diff = np.diff(quad, axis=1)
        rectangle[1] = quad[np.argmin(diff)]  # top-right
        rectangle[3] = quad[np.argmax(diff)]  # bottom-left

        return rectangle

    def _warp_rectangle(
        self, preprocessed_image: MatLike, ordered_corners: np.ndarray
    ) -> MatLike:
        """Flatten a detected card into a straight-on CARD_WIDTH_PX x CARD_HEIGHT_PX rectangle."""
        # describe the 4 corners of a flat, upright card at our target size
        dst = np.array(
            [
                [0, 0],
                [CARD_WIDTH_PX - 1, 0],
                [CARD_WIDTH_PX - 1, CARD_HEIGHT_PX - 1],
                [0, CARD_HEIGHT_PX - 1],
            ],
            dtype="float32",
        )

        # compute the matrix that maps the detected corners onto that flat rectangle
        M = cv2.getPerspectiveTransform(ordered_corners, dst)

        # apply it and return the warped, upright card
        return cv2.warpPerspective(
            preprocessed_image, M, (CARD_WIDTH_PX, CARD_HEIGHT_PX)
        )

    def detect_cards(self) -> list[MatLike]:
        # final list of processed card candidates
        card_candidates = []

        # resize the image
        resized_image = self._resize_image()

        # apply CLAHE normalisation to image
        preprocessed_image = self._preprocess_image(resized_image)

        # find contours
        contours = self._find_contours(preprocessed_image)

        # filter for quadrilater shapes that resemble a magic card's aspect ratio
        quads = self._filter_contours_to_cardlike_quadrilaterals(contours)

        # isolate each candidate shape in the preprocessed image and warp straight-on
        for q in quads:
            rectangle = self._order_points(q)
            warped_rectangle = self._warp_rectangle(preprocessed_image, rectangle)
            card_candidates.append(warped_rectangle)

        return card_candidates
