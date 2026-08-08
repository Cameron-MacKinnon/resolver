import cv2
import imagehash
from cv2.typing import MatLike
from PIL import Image

# imagehash.phash's default hash_size=8 (64 bits) doesn't have enough entropy
# to keep tens of thousands of cards well-separated - verified empirically
# against the real card corpus, where several unrelated cards tied for the
# closest match at hash_size=8, but uniquely separated at hash_size=16
PHASH_SIZE = 16


class Hasher:
    def __init__(self, image: MatLike) -> None:
        self.image = image

    def _convert_to_pil(self, image: MatLike) -> Image.Image:
        """Convert an OpenCV image (BGR channel order) into a PIL image (RGB), which imagehash requires."""
        # convert from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert RGB image to PIL and return
        pil_image = Image.fromarray(rgb_image)
        return pil_image

    def compute_hash(self) -> imagehash.ImageHash:
        """Compute the image's perceptual hash (phash). We use phash since it remains fairly stable
        across minor variations of the same image
        """
        # convert image to format that phash can work with
        pil_image = self._convert_to_pil(self.image)

        # compute perceptual hash and return
        p_hash = imagehash.phash(pil_image, hash_size=PHASH_SIZE)
        return p_hash
