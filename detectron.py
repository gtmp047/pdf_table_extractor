import uuid
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

from detectron2.utils.visualizer import ColorMode

ROOT_DIR = "./"
ipdir = ROOT_DIR + "nswtable_input/image/"
opdir = ROOT_DIR + "results_nswtable/"


def predict(im, item):
    fileName = item
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print(outputs["instances"].pred_boxes.tensor.numpy())
    path = "/root/images/"
    path1 = "/root/tblImg/"
    cv2.imwrite(path1 + fileName + ".png", v.get_image()[:, :, ::-1])
    boxes = {}

    file = os.path.join(path, fileName)
    try:
        f = os.makedirs(file, exist_ok=True)
        print("Directory '%s' created " % file)
    except OSError as error:
        print("cannot create" % directory)
    i = 1
    coords = []
    for coordinates in outputs["instances"].to("cpu").pred_boxes:

        coordinates_array = []
        for k in coordinates:
            coordinates_array.append(int(k))
        boxes[uuid.uuid4().hex[:].upper()] = coordinates_array
        coords.append(coordinates_array)

    for k, v in boxes.items():

        crop_img = im[v[1]:v[3], v[0]:v[2], :]
        # print(v[1],v[3], v[0],v[2])
        # cv2_imshow(crop_img)
        crop_width, crop_height = crop_img.shape[0], crop_img.shape[1]
        if crop_width > crop_height:
            img_rot = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)

            # ------for naming the images------#v[1]=y,v[3]=y+h, v[0]=x,v[2]=x+w
            margin = 0

            ymin = max(v[1] - margin, 0)
            ymax = v[3] + margin
            xmin = max(v[0] - margin, 0)
            xmax = v[2] + margin
            # print(ymin,ymax,xmin,xmax)
            cv2.imwrite(
                file + '/' + str(i) + '_' + str(xmin) + '_' + str(ymin) + '_' + str(xmin) + '_' + str(ymax) + '_' + str(
                    xmax) + '_' + str(ymin) + '_' + str(xmax) + '_' + str(ymax) + '.png', img_rot)
            i = i + 1

    return outputs


dirs = os.listdir(ipdir)

for item in dirs:
    if os.path.isfile(ipdir + item):
        im = cv2.imread(ipdir + item)
        print(item)
        f, e = os.path.splitext(ipdir + item)
        # width,height = im.shape[1],im.shape[0]
        item = item[:-4]
        predict(im, item)