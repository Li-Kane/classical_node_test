import cv2
import numpy as np

from .cvutils import get_bb_of_largest_cc


def width_ratio(robot_type_mask: np.ndarray) -> float:
    rect = get_bb_of_largest_cc(robot_type_mask)
    (_, _), (width, height), angle = rect
    if abs(angle) > 45:
        width, height = height, width
    mask_width = robot_type_mask.shape[1]
    return width / mask_width


def do_template_matching(img: np.ndarray, template: np.ndarray, shrinking_factor: int,
                         match_type: int = cv2.TM_CCOEFF_NORMED) -> float:
    shrank = template[shrinking_factor:-shrinking_factor, shrinking_factor:-shrinking_factor]
    res = cv2.matchTemplate(img, shrank, match_type)
    return np.max(res)
