import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import mmseg
print(mmseg.__version__)

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv import Config
from mmseg.apis import set_random_seed

config_file = 'config.py'
checkpoint_file = '../checkpoints/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth'#vit_tiny_p16_384_20220308-cce8c795.pth'

model = init_segmentor(config_file, checkpoint_file, device='cpu')
#for m in model.modules():
#    print(m)
model.decode_head.num_classes = 21
print(model.decode_head.num_classes)


#img = '../demo/demo.png'
#result = inference_segmentor(model, img)

#show_result_pyplot(model, img, result) #get_palette('cityscapes'))
