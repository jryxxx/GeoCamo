# Width rounding used by the YOLOv3 model parser.

import math

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor
