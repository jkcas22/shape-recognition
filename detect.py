"""
Script to run object detection on images.
`--input` can take the path either an image or a directory containing images.
"""

import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame 

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.transforms import infer_transforms, resize

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input image (one image or a folder path)',
    )
    parser.add_argument(
        '-o', '--output', 
        help='folder path to output results image',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default=None,
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', 
        default=0.3, 
        type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show',  
        action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=None,
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-n', '--num-images',
        dest='num_images', 
        default=3,
        type=int,
        help='process only the first n images in folder path'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    parser.add_argument(
        '--square-img',
        dest='square_img',
        action='store_true',
        help='whether to use square image resize, else use aspect ratio resize'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['data'] is not None:
        with open(args['data']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']
    OUT_DIR = args['output']

    # Load the pretrained model
    if args['weights'] is None:
        # If the config file is still None, 
        # then load the default one for COCO.
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['data']['NC']
            CLASSES = checkpoint['data']['CLASSES']
        try:
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    if args['input'] == None:
        DIR_TEST = data_configs['image_path']
        test_images = collect_all_images(DIR_TEST)
    else:
        DIR_TEST = args['input']
        test_images = collect_all_images(DIR_TEST)

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    N = args['num_images']
    n = len(test_images)
    if n>N: n=N
    for i in range(n):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        orig_image = cv2.imread(test_images[i])
        frame_height, frame_width, _ = orig_image.shape
        if args['imgsz'] != None:
            RESIZE_TO = args['imgsz']
        else:
            RESIZE_TO = frame_width
        # orig_image = image.copy()
        image_resized = resize(
            orig_image, RESIZE_TO, square=args['square_img']
        )
        image = image_resized.copy()
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        boxes = DataFrame(outputs[0]['boxes'].data.numpy(),columns=['x0','y0','x1','y1'])
        scores = DataFrame({'score':outputs[0]['scores'].data.numpy()})
        labels = DataFrame({'class':[CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]})
        result = pd.concat([labels,scores,boxes],axis=1)

        print(f"{DIR_TEST}/{image_name}.jpg --> {OUT_DIR}/{image_name}.jpg")
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            orig_image = inference_annotations(
                outputs, 
                detection_threshold, 
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )
            cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
            result.to_csv(f"{OUT_DIR}/{image_name}.csv",index=None)
            if args['show']:
                image = plt.imread(f"{OUT_DIR}/{image_name}.jpg")
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                plt.axis('off')
                plt.show()
                print('\n')
    _= cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_opt()
    main(args)