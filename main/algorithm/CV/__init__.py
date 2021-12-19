__version__ = "0.0.1"

from main.algorithm.CV.drawing_style import transform_to_painting
from main.algorithm.CV.scan import scanning
from main.algorithm.CV.sift import sift_matching
from main.algorithm.CV.hrr import reconstruct
from main.algorithm.CV.face_detection import detect_face
from main.algorithm.CV.stitcher import stitching
from main.algorithm.CV.ocr import ocr_val, ocr_print
from main.algorithm.CV.basic import equalizeHist, OSTU_split
from main.algorithm.CV.utils import url_imread, show_image, resize