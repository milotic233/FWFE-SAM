import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from samnet import SAMNET
from data import get_loader, test_dataset  # 自定义的数据加载器
from utils import clip_gradient  # 自定义梯度裁剪  # 自定义 SSIM Loss
import torch.backends.cudnn as cudnn
import logging
import time 
import math
import cv2
# torch.autograd.set_detect_anomaly(True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from data import test_dataset  # 自定义的数据加载模块

# 设置设备
torch.cuda.set_device(1)

# 解析输入参数
class opt:
    test_model = "/data/lxy-workspace/sam3/checkpoints/kanmodel-3-new-exchange-3branch/SAMNET_best_trainable.pth"  # 模型路径
    test_data_root = "/data/lxy-workspace/samsod/VT/"  # 测试数据根目录
    maps_path = "./results/"  # 结果保存路径
    testsize = 504  # 测试图像尺寸


# 加载模型
print("Loading SAMNET model...")
model = SAMNET(checkpoint_path=opt.test_model)
# checkpoint = torch.load(opt.test_model)
# state_dict = checkpoint 
# missing, unexpected = model.load_state_dict(state_dict, strict=True)
# print(f"Missing keys: {len(missing)}")
# print(f"Unexpected keys: {len(unexpected)}")

model = model.cuda()
model.eval()

# 测试集
test_sets = ["VT5000/Test","VT1000","VT821"]

for dataset in test_sets:
    save_path = os.path.join(opt.maps_path, dataset)
    save_path_mask = os.path.join(save_path, "mask")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_mask):
        os.makedirs(save_path_mask)

    dataset_path = os.path.join(opt.test_data_root, dataset)
    test_loader = test_dataset(dataset_path, opt.testsize)
    with torch.no_grad():
    # 开始测试
        for i in range(test_loader.size):
            vis_image, inf_image, gt, (H, W), name = test_loader.load_data()
            vis_image, inf_image = vis_image.cuda(), inf_image.cuda()
            shape = (W, H)

            # 前向传播
            _,_,_,pred = model(vis_image, inf_image)
            pred = pred.sigmoid().cpu().numpy().squeeze()  # 将预测结果转为 numpy 格式
            # pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 归一化到 [0, 1]

            # 保存结果
            result_path = os.path.join(save_path, name)
            cv2.imwrite(result_path, (pred * 255).astype(np.uint8))
            print(f"Saved result to: {result_path}")



print("Test Done!")