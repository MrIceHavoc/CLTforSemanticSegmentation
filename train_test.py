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

DATASETS_DIR = '../'
VOC_DATASET = os.path.join(DATASETS_DIR, 'VOCdevkit/VOC2012/')

def info_nce_loss(features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
        
def main(checkpoint, config_file, dataset, mode):
    print(dataset)
    if mode == 'train':
        cfg = mmcv.Config.fromfile(config_file)
        cfg.dataset_type = 'PascalVOCDataset'
        cfg.dataset_root = dataset
        cfg.model.decode_head.num_classes = 21

        cfg.data.train.type = cfg.dataset_type
        cfg.data.train.data_root = cfg.data_root
        cfg.data.train.img_dir = os.path.join(dataset, 'JPEGImages/')
        cfg.data.train.ann_dir = os.path.join(dataset, 'Annotations/')
        cfg.data.train.pipeline = cfg.train_pipeline
        cfg.data.train.split = os.path.join(dataset, 'ImageSets/Segmentation/train.txt')
        cfg.load_from = checkpoint

        cfg.work_dir = './work_dirs/clt_segmentation'
        cfg.seed = 42
        cfg.device = get_device()

        datasets = [build_dataset(cfg.data.train)]
        model = build_segmentor(cfg.model)
        model.CLASSES = datasets[0].CLASSES
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
    elif mode == 'test':
        cfg = mmcv.Config.fromfile(config_file)
        cfg.dataset_type = 'PascalVOCDataset'
        cfg.dataset_root = dataset
        cfg.model.decode_head.num_classes = 21

        cfg.data.test.type = cfg.dataset_type
        cfg.data.test.data_root = cfg.data_root
        cfg.data.test.img_dir = os.path.join(dataset, 'JPEGImages/')
        cfg.data.test.ann_dir = os.path.join(dataset, 'Annotations/')
        cfg.data.test.pipeline = cfg.test_pipeline
        cfg.data.test.split = os.path.join(dataset, 'ImageSets/Segmentation/val.txt')
        cfg.load_from = checkpoint

        model = build_segmentor(cfg.model)
        cfg.work_dir = './work_dirs/clt_segmentation'
        cfg.seed = 42
        cfg.device = get_device()

        datasets = [build_dataset(cfg.data.train)]
        model = build_segmentor(cfg.model)
        model.CLASSES = datasets[0].CLASSES
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
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
    parser.add_argument('--model', default=os.path.join(cwd, '/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth'))
    parser.add_argument('--config', default=cwd + '/config.py')
    parser.add_argument('--dataset', type=pathlib.Path, default=cwd + VOC_DATASET)
    parser.add_argument('--mode', choices=['train', 'test', 'inference'], default='train')
    args = vars(parser.parse_args())
    main(args['model'], args['config'], args['dataset'], args['mode'])
