import glob
import random

all_imgs = glob.glob("/z/bulk/store_qtracking/training_data/images/*.jpg")
random.shuffle(all_imgs)
total_img = len(all_imgs)
split_percent = 0.9
check_split = split_percent*total_img
train_file = open("/z/bulk/store_qtracking/training_data/train.txt", "w")
val_file = open("/z/bulk/store_qtracking/training_data/val.txt", "w")
for i in range(total_img):
    if i > check_split:
        val_file.write(all_imgs[i]+"\n")
    else:
        train_file.write(all_imgs[i]+"\n")
train_file.close()
val_file.close()
