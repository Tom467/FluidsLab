
import cv2
import numpy as np


def crop(x1: int = 0, x2: int = None, y1: int = 0, y2: int = None):
    return lambda image: image[x1: x2, y1: y2]


def canny(threshold1: int = 100, threshold2: int = 200):
    return lambda image: cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)


if __name__ == "__main__":
    a = [0, 1, 2, 3]
    print(a[0:None])
