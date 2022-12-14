import mmcv
import numpy as np
import os

from PIL import Image

from ..builder import DATASOURCES
from .base import BaseDataSource


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def get_samples(idx_file, img_folder, label_folder, extensions):
    """Make dataset by walking all images under a root.

    Args:
        idx_file (string): file which indicates the images and labels names
        img_folder (string): directory for the images
        label_folder (string): directory for the labels
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    images = []
    labels = []
    print(os.getcwd())

    for (_, _, files) in os.walk(img_folder, topdown=True):
        for file in files:
            if has_file_allowed_extension(file, extensions) and os.path.splitext(file)[0] in open(idx_file).read():
                images.append(file)
        
    for (root, _, files) in os.walk(label_folder, topdown=True):
        for file in files:
            if has_file_allowed_extension(file, extensions) and os.path.splitext(file)[0] in open(idx_file).read():
                labels.append(os.path.join(root, file))

    samples = []
    for img, label in zip(images, labels):
        label = Image.open(label)
        samples.append((img, label))

    return samples


@DATASOURCES.register_module()
class VOCSegmentation(BaseDataSource):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
    """

    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"
    _INPUT_FILE_EXT = ".jpg"

    def load_annotations(self):
        assert self.ann_file is not None
        if not isinstance(self.ann_file, list):
            self.ann_file = [self.ann_file]

        data_infos = []
        for ann_file in self.ann_file:
            with open(ann_file, 'r') as f:
                label_dir = os.path.split(self.data_prefix)[0] + self._TARGET_DIR
                samples = get_samples(ann_file, self.data_prefix, label_dir, extensions=(self._INPUT_FILE_EXT, self._TARGET_FILE_EXT))

            self.samples = samples

            for i, (filename, gt_label) in enumerate(self.samples):
                #print(gt_label)
                #print(filename)
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                info['idx'] = int(i)
                data_infos.append(info)

        return data_infos
