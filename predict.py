#! /usr/bin/env python

import time
import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np


def _main_(args):
    config_path = args.conf
    input_path = args.input
    output_path = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    # do detection on an image or a set of images
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (
        inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    # the main loop
    times = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image_path)
        start = time.time()
        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [
            image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
        print('Elapsed time = {}'.format(time.time() - start))
        times.append(time.time() - start)
        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh)

        # write the image with bounding boxes to file
        cv2.imwrite(output_path + image_path.split('/')
                    [-1], np.uint8(image))

    file = open(args.output + '/time.txt', 'w')
    file.write('Tiempo promedio:' + str(np.mean(times)))
    file.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument(
        '-i', '--input', help='path to an image or a directory of images')
    argparser.add_argument(
        '-o', '--output', default='output/', help='path to output directory')

    args = argparser.parse_args()
    _main_(args)
