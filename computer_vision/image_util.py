import numpy as np
import glob
import cv2


def error(a, b):
    if a == 0:
        if b == 0:
            return 100000
        return abs(a - b) / b
    return abs(a - b) / a


def read_image_folder(folder_path, file_extention='.tif', read_color=False):
    # TODO add crop region
    files = glob.glob(folder_path + '/*' + file_extention)
    images = []
    for i in files:
        if read_color:
            img = cv2.imread(i)
        else:
            img = cv2.imread(i, 0)
        images.append(img)
    images = np.stack(images, 0)
    return images


def write_video(output_file, images, framerate=20, color=False, fourcc=cv2.VideoWriter_fourcc(*'mp4v')):
    video = cv2.VideoWriter(output_file, fourcc, framerate, (images.shape[2], images.shape[1]), color)
    for i in images:
        video.write(i)
    video.release()


def animate_images(images, wait_time=10, wait_key=False):
    window_name = 'image'
    for i, image in enumerate(images):
        cv2.setWindowTitle('window', str(i))
        cv2.imshow(window_name, image)

        # Press Q on keyboard to exit
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
        if wait_key:
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_contours(img, threshold1=100, threshold2=200):
    img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    return cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


def simple_contour_match(contours, target_contour):
    # TODO Try using two contours combined to improve matching accuracy
    target_position = np.mean(target_contour[:, :, 0]), np.mean(target_contour[:, :, 1])
    err = 1000000
    for con in contours:
        shape_err = cv2.matchShapes(target_contour, con, 1, parameter=0)
        length_error = error(con.shape[0], target_contour.shape[0])
        position_error = error(np.mean(target_contour[:, :, 0]), np.mean(con[:, :, 0])) + error(
            np.mean(target_contour[:, :, 1]), np.mean(con[:, :, 1]))
        if shape_err + length_error + position_error < err:
            err = shape_err + length_error + position_error
            matching_contour = con
    return matching_contour


def generate_gif(images_location, output_file):
    img_array = []
    for filename in glob.glob(images_location):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('complete')