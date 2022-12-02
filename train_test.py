#TODO: Implemement Training and Test pipeline with visualization
import mmcv
import matplotlib.pyplot as plt
import argparse
import os
import pathlib
import numpy as np
from PIL import Image
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

img_dir = 'JPEGImages/'

if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(prog='CLT', description="Train/Test Contrastive Learning Transformer for Semantic Segmentation.")
    parser.add_argument('--model', default=cwd + '/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth')
    parser.add_argument('--config_file', default=cwd + '/config.py')
    parser.add_argument('--dataset', type=pathlib.Path, default=cwd + '/VOCdevkit/VOC2012/')
    parser.add_argument('--mode', choices=['train', 'test', 'inference'], default='train')
    args = vars(parser.parse_args())
    model = init_segmentor(args['config_file'], args['model'])
    if args['mode'] == 'train':
        s = 0
    elif args['mode'] == 'test':
        s = 1
    elif args['mode'] == 'inference':
        for file in mmcv.scandir(os.path.join(args['dataset'], img_dir), suffix='.jpg'):
            img = mmcv.imread(file)
            result = inference_segmentor(model, img)
            show_result_pyplot(model, img, result, get_palette('VOCdevkit'))