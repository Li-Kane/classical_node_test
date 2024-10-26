import os
from dataclasses import dataclass
from enum import Enum
from typing import List

import cv2
import numpy as np

from .classicalcv.colorutils import classify_color, threshold_by_color
from .classicalcv.lightbarutils import get_light_bar_pair, unwarp_img_from_lbs
from .classicalcv.panelclasses import PanelColor
from .classicalcv.sizeutils import do_template_matching
from vision_msgs.msg import Detection2DArray

directory = '/home/kane-li/Documents/GitHub/TR-CV-2024/src/TR-Binary-Classical-Node/classical_node/templates'
filelist = os.listdir(directory)

templates = {file: cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE) for file in filelist}

teamcolor = PanelColor.RED

min_consensus = 0.1



class ShootEnum(Enum):
    Unsafe = 'Own team or base detected, unsafe to shoot'
    SafeToIgnore = 'Low conf or purple, safe to ignore'
    Shoot = 'Opp team detected, safe to shoot'


def check_image(image: np.ndarray) -> ShootEnum:
    try:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        color = classify_color(ycrcb)
    except BaseException:
        print('Fail to cvt/ classify')
        return ShootEnum.SafeToIgnore
    mask = threshold_by_color(ycrcb, color)
    try:
        lbpair = get_light_bar_pair(mask)
    except BaseException:
        print('Fail to lb')
        return ShootEnum.SafeToIgnore
    unwarped = unwarp_img_from_lbs(ycrcb[:, :, 0], lbpair)
    _, thresh = cv2.threshold(unwarped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    values: dict[str, float] = {}

    for file, image in templates.items():
        val = do_template_matching(thresh, image, 3)
        values[file] = val

    if max(values.values()) < .2:
        return ShootEnum.SafeToIgnore

    if max(values, key=values.get) == 'largebase_middle_template.png':
        return ShootEnum.Unsafe

    return ShootEnum.Unsafe if color == teamcolor else ShootEnum.Shoot

def decide_to_shoot(opinions: List[ShootEnum]) -> bool:
    if ShootEnum.Unsafe in opinions:
        return False
    consensus = opinions.count(ShootEnum.Shoot) > len(opinions) * min_consensus
    return consensus