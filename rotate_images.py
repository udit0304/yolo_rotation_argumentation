import numpy as np
import cv2
from glob import glob
import os
import argparse

# parsing
parser = argparse.ArgumentParser()
parser.add_argument("dataset_input",
                    help="directory containing data you want to rotate.")
parser.add_argument("-o",
                    dest="dataset_output",
                    help="directory to store generated data. this directory will be made automatically.",
                    default="data_rotational")
parser.add_argument("-t",
                    dest="time_interval",
                    help="time interval to control speed of displaying images.",
                    default=1,
                    type=int)
parser.add_argument("-r",
                    dest="ratio",
                    help="ratio for ignoring bounding box near the edges of image.",
                    default=0.8,
                    type=float)
parser.add_argument("-a",
                    dest="angle_interval",
                    help="angle interval for rotating.",
                    default=30,
                    type=int)
parser.add_argument("-s",
                    dest="show_image",
                    action="store_true",
                    help="instead of saving data, showing images with bounding boxes without saving.",
                    default=False)

args = parser.parse_args()
dataset_input = args.dataset_input
dataset_output = args.dataset_output
time_interval = args.time_interval
ratio = args.ratio
angle_interval = args.angle_interval
show_image = args.show_image

dir_input_image = dataset_input
dir_output_image = dataset_output

data_folder = [8] #range(9)
data_subfolder = range(30, 34)
root_folder = "/z/tmp/code/data/"
for i in data_folder:
    for j in data_subfolder:
        data_path = root_folder + str(i) + "/" + str(j)
        image_names = sorted(glob(data_path + "/*jpg"))
        print("# of images: %d" % len(image_names))

        for image_name0 in image_names:
            print(image_name0)
            image0 = cv2.imread(image_name0)
            height_image, width_image = image0.shape[:2]
            angle = angle_interval
            image_name = image_name0
            # print(image_name)
            if angle == 0.:
                image = np.array(image0)
                nW = width_image
                nH = height_image
            else:
                center = int(width_image / 2), int(height_image / 2)
                scale = 1.
                matrix = cv2.getRotationMatrix2D(center, angle, scale)
                cos = np.abs(matrix[0, 0])
                sin = np.abs(matrix[0, 1])
                # compute the new bounding dimensions of the image
                nW = width_image  # int((height_image * sin) + (width_image * cos))
                nH = height_image  # int((height_image * cos) + (width_image * sin))
                # adjust the rotation matrix to take into account translation
                matrix[0, 2] += (nW / 2) - center[0]
                matrix[1, 2] += (nH / 2) - center[1]
                # image = cv2.warpAffine(image0,matrix,(width_image,height_image),borderMode=cv2.BORDER_REPLICATE)
                image = cv2.warpAffine(image0, matrix, (nW, nH))
            image_annotated = np.array(image)

            if not show_image:
                cv2.imwrite(image_name, image)
            else:
                cv2.imshow("test", image_annotated)

            if cv2.waitKey(time_interval) & 0xff == ord('q'):
                quit()



