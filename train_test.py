import mmcv
import matplotlib.pyplot as plt
import argparse
import os
import pathlib
import numpy as np
from PIL import Image
from mmseg.apis import train_segmentor, inference_segmentor, init_segmentor, show_result_pyplot, set_random_seed
from mmseg.core.evaluation import get_palette
from mmseg.utils import get_device
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
import torch

DATASETS_DIR = '/mmsegmentation/data/'

def main(checkpoint, config_file, dataset, mode):
    if mode == 'train':
        cfg = mmcv.Config.fromfile(config_file) 
        cfg.model.decode_head.num_classes = 21
        cfg.model.auxiliary_head.num_classes = 21
        cfg.device = get_device()
        model = build_segmentor(cfg.model)
        datasets = [build_dataset(cfg.data.train)]
        model.CLASSES = datasets[0].CLASSES
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
    elif mode == 'test':
        cfg = mmcv.Config.fromfile(config_file)
    elif mode == 'inference':
        model = init_segmentor(config_file, checkpoint)
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