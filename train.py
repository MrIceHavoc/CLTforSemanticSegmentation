import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

from vision_transformer import VisionTransformer
from swin_transformer import SwinTransformer

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import tqdm
import evaluate

#import datasets
import numpy as np
#from datasets import load_dataset

import pandas #Add import
import csv #Add import
#from datasets import Dataset, DatasetDict #Add import

#import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import SwinConfig, SwinModel, AutoImageProcessor, SegformerForSemanticSegmentation, SegformerDecodeHead, SegformerConfig, SegformerModel, Trainer, SegformerFeatureExtractor
from datasets import load_dataset
from transformers import TrainingArguments

from transformers import AutoModel
class PosModel(nn.Module):
    def __init__(self):
        super(PosModel, self).__init__()

        self.base_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        #(dense2): Linear(in_features=1024, out_features=256, bias=True)
        #      (dropout): Dropout(p=0.0, inplace=False)
        self.linear = nn.Linear(768, 256) # output features from bert is 768 and 2 is ur number of labels
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(0.2)
        self.decode_head = SegformerDecodeHead(SegformerConfig())
        self.decode_head.classifier = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, pixel_values, labels):#attn_mask
        print("!!!!!!!!!!!!",self.base_model(pixel_values).last_hidden_state.shape)
        outputs = self.base_model(pixel_values).last_hidden_state#, attention_mask=attn_mask
        print(outputs)
        # You write you new head here
        outputs = self.linear(outputs)
        outputs = self.linear2(outputs)
        outputs = self.linear3(outputs)
        outputs = self.dropout(outputs[0])
        outputs = self.decode_head(outputs)

        return outputs

model = PosModel()
for param in model.base_model.parameters():
    param.requires_grad = False#print(model)
#decode_head = SegformerDecodeHead(SegformerConfig())
#decode_head.classifier = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
#print(decode_head)
feature_extractor = SegformerFeatureExtractor()

def train_transforms(example_batch):
    #print("EXAMPLE BATCH", example_batch)
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = feature_extractor(images, labels)
    #print("INPUTS", inputs)#del dict[key]
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = feature_extractor(images, labels)
    return inputs

#dataset = load_dataset('pascal_voc/pascal_voc.py', 'voc2012_segmentation')
dataset = load_dataset("fuliucansheng/pascal_voc", 'voc2012_segmentation')

data_files = {"train": 'data/en_de/train_en_de.json', "test": 'data/en_de/test_en_de.json', "validation": 'data/en_de/validation_en_de.json'}#train_filtered_baseline2
#raw_datasets = load_dataset("json",
                            #data_files=data_files,
                            #cache_dir='./data_cache')

train_ds = dataset['train']#[:80%]#load_dataset("fuliucansheng/pascal_voc", 'voc2012_segmentation', split=['train[:80%]'])#("nateraw/pascal-voc-2012"))
train_ds = train_ds.rename_column('image', 'pixel_values')
train_ds = train_ds.rename_column('object_gt_image', 'label')#class_gt_image
#train_ds = train_ds.remove_columns(['id', 'height', 'width', 'class_gt_image', 'object_gt_image'])
#test_ds = dataset['train']#[-20%:]#load_dataset("fuliucansheng/pascal_voc", 'voc2012_segmentation', split=['train[-20%:]'])
print("BEFORE SET TRANSFORM", train_ds['pixel_values'])
train_ds.set_transform(train_transforms)
print("AFTER SET TRANSFORM", train_ds['pixel_values'])
train_ds = train_ds.remove_columns(['classes', 'height', 'width', 'class_gt_image'])
#test_ds.set_transform(val_transforms)
#print("TRAIN", train_ds)
#print("TEST", test_ds)
epochs = 50
lr = 0.00006
batch_size = 2
training_args = TrainingArguments(
    "test",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #save_total_limit=1,
    evaluation_strategy='no',
    save_strategy='no',
    #eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=False,
    #remove_unused_columns=False,
)

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    # currently using _compute instead of compute
    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
    metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=feature_extractor.reduce_labels,
        )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    #eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

#model.to('cuda')
#dataset = load_dataset("huggingface/cats-image")

#image = dataset["test"]["image"][0]

#feature_extractor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
"""
model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

configuration = SegformerConfig()

# Initializing a model from the nvidia/segformer-b0-finetuned-ade-512-512 style configuration

model2 = SegformerForSemanticSegmentation(configuration)
print(model)

decode_head = SegformerDecodeHead(configuration)
print(decode_head)"""

"""
dataset = load_dataset("nateraw/pascal-voc-2012")
# Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
back_state_dict = torch.load("../checkpoints/checkpoint_best.pth", map_location=torch.device('cpu'))

configuration = SwinConfig()

# Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration

model = SwinModel(
            configuration,
            )

# Accessing the model configuration

configuration = model.config"""
