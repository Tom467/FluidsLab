import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def error(a, b):
    if a == 0:
        if b == 0:
            return 100000
        return abs(a-b)/b
    return abs(a-b)/a


images_location = "C:/Users/truma/Documents/MATLAB/28679_1_89/*.tif"
image_files = glob.glob(images_location)
images = []
for image_file in image_files[2000:2020]:
    img = cv2.imread(image_file, flags=0)
    img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=0, threshold2=20)
    contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    img_copy = img.copy()
    long_contours = []
    lengths = []
    for con in contours:
        (x, y), radius = cv2.minEnclosingCircle(con)
        hull = cv2.convexHull(con)
        # if con.shape[0] > 200 and con.shape[0] < 250:
        if error(cv2.contourArea(con), cv2.contourArea(hull)) < .1 and con.shape[0] > 100:
            if error(x, 525) < .01 and error(y, 344) < .01:
                print(x, y)
                lengths.append(con.shape[0])
                long_contours.append(con)
    contour = cv2.drawContours(image=img, contours=long_contours, contourIdx=-1, color=(255, 255, 255),
                               thickness=2, lineType=cv2.LINE_AA)
    images.append(img)

window_name = 'image'
for image in images:
    cv2.imshow(window_name, image)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()