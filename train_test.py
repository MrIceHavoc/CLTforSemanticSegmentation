import mmcv
import matplotlib.pyplot as plt
import argparse
import os
import pathlib
import numpy as np
from PIL import Image
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

DATASETS_DIR = '/mmsegmentation/data/'

def main(model, config_file, dataset, mode):
    model = init_segmentor(config_file, mode)
    if mode == 'train':
        s = 0
    elif mode == 'test':
        s = 1
    elif mode == 'inference':
        img_dir = 'JPEGImages/'
        for file in mmcv.scandir(os.path.join(dataset, img_dir), suffix='.jpg'):
            img = mmcv.imread(file)
            result = inference_segmentor(model, img)
            show_result_pyplot(model, img, result, get_palette('voc'))

if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(prog='CLT', description="Train/Test Contrastive Learning Transformer for Semantic Segmentation.")
    parser.add_argument('--model', default=cwd + '/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth')
    parser.add_argument('--config', default=cwd + '/config.py')
    parser.add_argument('--dataset', type=pathlib.Path, default=cwd + DATASETS_DIR + 'VOCdevkit/VOC2012/')
    parser.add_argument('--mode', choices=['train', 'test', 'inference'], default='train')
    args = vars(parser.parse_args())
    main(args['model'], args['config'], args['dataset'], args['mode'])