#!/usr/bin/env python3

import os
import shutil
import Augmentor


def genrate_images(source, output, number):
    p = Augmentor.Pipeline(source, output_directory=output)
    p.flip_random(probability=1)
    p.flip_left_right(probability= 1.0)
    p.flip_top_bottom(probability= 1.0)
    p.crop_random(probability=1.0, percentage_area=0.90)
    p.rotate_random_90(probability=0.9)
    p.rotate180(probability=0.8)
    p.rotate270(probability=0.6)
    p.flip_random(probability=1)
    p.rotate(probability=0.5, max_left_rotation=4, max_right_rotation=4)
    p.shear(probability=0.6, max_shear_left=8, max_shear_right=8 )
    p.zoom(probability=0.8, min_factor=1, max_factor=1.3)
    p.sample(number)
    shutil.rmtree(output+"/0")

cates = os.listdir("/home/bcri/train_data2/data143")
cates_images = {}
for cat in cates:
    cates_images[cat] = []
    cates_images[cat].extend(os.listdir("/home/bcri/train_data2/data143/"+ cat))
    to_generate = 80000 - len(cates_images[cat])
    cates_images[cat].extend([None for _ in range(to_generate)])


for file in cates:
    os.mkdir("/home/bcri/train_data2/RDD1_data03/"+file)
    generate = 0
    for i in range(0, 80000):
        if cates_images[file][i] is not None:
            shutil.copy("/home/bcri/train_data2/data143/"+file+"/"+cates_images[file][i], "/home/bcri/train_data2/RDD1_data03/"+file)
        else:
            generate +=1 
    if generate !=0:
        genrate_images("/home/bcri/train_data2/data143/"+file, "/home/bcri/train_data2/RDD1_data03/"+file, generate )
    print(file, generate)



