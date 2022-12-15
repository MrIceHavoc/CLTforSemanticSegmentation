import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

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
from transformers import SwinConfig, SwinModel, SegformerPreTrainedModel, SegformerForSemanticSegmentation, SegformerConfig, SegformerModel, Trainer, SegformerFeatureExtractor
from datasets import load_dataset
from transformers import TrainingArguments

from transformers import AutoModel


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, decoder_hidden_size, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states = self.proj(hidden_states)
        return hidden_states


class PosDecoderHead(nn.Module):
    def __init__(self):
        super().__init__()
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same
        self.num_encoder_blocks = 1
        self.hidden_sizes = [256]
        self.decoder_hidden_size = 256
        self.classifier_dropout_prob = 0.2
        self.num_labels = 21
        mlps = []
        for i in range(self.num_encoder_blocks):
            mlp = SegformerMLP(self.decoder_hidden_size, input_dim=self.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=self.decoder_hidden_size * self.num_encoder_blocks,
            out_channels=self.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(self.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(self.classifier_dropout_prob)
        self.classifier = nn.Conv2d(self.decoder_hidden_size, self.num_labels, kernel_size=1)


    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )


            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            print("Enc hidden", encoder_hidden_state.shape)
            print("MLP: ", mlp)
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


class PosModel(nn.Module):
    def __init__(self):
        super(PosModel, self).__init__()

        self.base_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.linear = nn.Linear(768, 256) # output features from bert is 768 and 2 is ur number of labels
        #self.linear2 = nn.Linear(256, 128)
        #self.linear3 = nn.Linear(128, 32)
        self.proj = nn.Conv2d(2, 256, kernel_size=1, stride=1, padding=0)
        self.decode_head = PosDecoderHead()

    def forward(self, pixel_values, labels):#attn_mask
        batch_size = pixel_values.shape[0]
        
        outputs = self.base_model(pixel_values).last_hidden_state#, attention_mask=attn_mask
        print(outputs.shape)
        # You write you new head here
        outputs = self.linear(outputs)
        #outputs = self.linear2(outputs)
        #outputs = self.linear3(outputs)
        outputs = self.proj(outputs)
        outputs = outputs.reshape(batch_size, outputs.shape[0], outputs.shape[1], -1).permute(0, 3, 1, 2).contiguous()
        print(outputs.shape)
        #outputs = outputs.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outputs = self.decode_head(outputs)

        return outputs

model = PosModel()
for param in model.base_model.parameters():
    param.requires_grad = False#print(model)

print(model)
feature_extractor = SegformerFeatureExtractor()

def train_transforms(example_batch):
    print("EXAMPLE BATCH", example_batch)
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
#print("BEFORE SET TRANSFORM", train_ds['pixel_values'])
train_ds.set_transform(train_transforms)
#print("AFTER SET TRANSFORM", train_ds['pixel_values'])
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
    remove_unused_columns=False,
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
