import math

import torch

import torch.nn as nn
import torch.nn.functional as F
# from efficient_kan import KAN
from collections import OrderedDict
from efficient_kan import KAN
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision.transforms import v2
from sam3.model.vitdet import get_abs_pos
torch.cuda.set_device(0)
import gc
try:
    from timm.layers import DropPath, Mlp, trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import DropPath, Mlp, trunc_normal_
class KANAdapter(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.kan = KAN([
            in_features,
            int(in_features),
            in_features,
        ])
    
    def forward(self, t):
        # print(t.shape)
        return self.kan(t)    
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def dice_loss(inputs, targets):
    smooth = 1
    p = 2
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1-dice
class BiAdapterBlock(nn.Module):
    """
    复用同一个被冻结的 transformer block，
    用两个 KANAdapter 实现 t->rgb 与 rgb->t 的双向注入。
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block  # 共享复用
        # 显式冻结（尽管上游 image_encoder 已冻结，这里再次确保）
        for p in self.block.parameters():
            p.requires_grad = False

        self.features = block.attn.qkv.in_features
        self.kan_t2 = KANAdapter(self.features)  # 红外 -> 可见光
        self.kan_v2 = KANAdapter(self.features)  # 可见光 -> 红外

    def forward(self, rgb_feature, t_feature):
        # 假设输入为 [B, H, W, C]
        # 先做 KAN 双向注入（保持通道数不变）
        rgb_in = rgb_feature + self.kan_v2(t_feature)
        t_in   = t_feature + self.kan_t2(rgb_feature)
        # 同一个 block 先后作用于两条流（共享权重、无梯度更新）
        rgb_out = self.block(rgb_in)
        t_out = self.block(t_in)
        return rgb_out, t_out
class ABlock(nn.Module):
    """
    单分支 block 的包装，用于那些不需要 KAN 注入的层。
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, rgb_feature, t_feature):
        return self.block(rgb_feature), self.block(t_feature)
    
class SoftChannelSelect(nn.Module):
    """
    可微软通道选择，用 softmax(|x|/τ) 来选择重要通道。
    τ 越小 -> 越接近 hard top-k。
    """
    def __init__(self, tau: float = 1):
        super().__init__()
        self.tau = tau

    def forward(self, x):
        gate = torch.sigmoid(x / self.tau) 
        return x * gate
    
class GatedAdapter(nn.Module):
    """
    GatedAdapter（保留原结构，只将 topk 替换为可微软通道选择）
    """
    def __init__(self, in_dim, mid_dim, scale_value, init_option="lora"):
        super(GatedAdapter, self).__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.scale = scale_value

        # 下采样 / 上采样投影
        self.down_proj = nn.Linear(self.in_dim, self.mid_dim)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(self.mid_dim, self.in_dim)

        # *** 在这里替换为软通道选择 ***
        self.soft_select = SoftChannelSelect()

        # LoRA 风格初始化
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

        # 动态门控
        hidden_gate_dim = max(in_dim // 2, 16)
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_gate_dim),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, H, W, C]
        down = self.down_proj(x)
        down = self.non_linear_func(down)

        # ==== Soft Channel Selection（替代 topk） ====
        down = self.soft_select(down)

        up = self.up_proj(down)
        transformed = up * self.scale  # [B,H,W,C]

        # adaptive gate
        gate = self.gate(x)

        return gate * transformed

class SAMNET(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super().__init__()
        bpe_path = "/data/lxy-workspace/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        sam3 = build_sam3_image_model(bpe_path=bpe_path,checkpoint_path="/home/magus/.cache/modelscope/hub/models/facebook/sam3/sam3.pt", enable_inst_interactivity=True)
        self.sam = Sam3Processor(sam3)
        
        self.image_encoder = self.sam.model.backbone.vision_backbone
        # self.predictor = sam.model.inst_interactive_predictor
        self.mask_decoder = self.sam.model.inst_interactive_predictor.model.sam_mask_decoder
        self.prompt_encoder = self.sam.model.inst_interactive_predictor.model.sam_prompt_encoder

        # 冻结 backbone & prompt encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        # for param in self.predictor.parameters():
        #     param.requires_grad = False
        self.trunk = self.image_encoder.trunk
        # self.no_mem_embed=self.sam.model.inst_interactive_predictor.model.no_mem_embed
        # for name, module in self.image_encoder.named_children():
        #     print(f"Layer: {name} | Type: {type(module).__name__}")
        self.position_encoding = self.image_encoder.position_encoding
        self._bb_feat_sizes = [
            
            (144, 144),
            (72, 72),
            (36, 36),
        ]
        self.convs = self.image_encoder.convs
        self.sam2_convs = self.image_encoder.sam2_convs
        self.transform = self.sam.transform
        self.patch_embed = self.trunk.patch_embed
        # self.neck = self.image_encoder.neck
        # self.pos_embed_window = self.trunk.pos_embed_window
        self.pos_embed = self.trunk.pos_embed
        self.ln_pre = self.trunk.ln_pre
        self.ln_post = self.trunk.ln_post
        self.full_attn_ids = self.trunk.full_attn_ids
        self.rgb_adapters = nn.ModuleList()
        self.t_adapters = nn.ModuleList()
        num_blocks = len(self.trunk.blocks)  # 通常为 48

        for i in range(num_blocks):
            self.rgb_adapters.append(GatedAdapter(1024, 32, 0.5))
            self.t_adapters.append(GatedAdapter(1024, 32, 0.5))


        # 包装 block：stage 头层 + 第 0 层用 BiAdapter（KAN），其余用 ABlock
        self.blocks = nn.ModuleList([
           # BiAdapterBlock(block) if  (i - 1) in self.full_attn_ids or i == 0  else ABlock(block)
            
        ])
        for i, block in enumerate(self.trunk.blocks):
            if  (i - 1) in self.full_attn_ids or i == 0:
                self.blocks.append(BiAdapterBlock(block))
            else: self.blocks.append(ABlock(block))
        # self.blocks = nn.ModuleList(
        #     [
        #         ABlock(block)
        #         for i, block in enumerate(self.trunk.blocks)
        #     ]
        # )

        # # FPN 多尺度融合 conv（RGB+T->融合）
        self.fuse_convs =nn.Conv2d(2048, 1024, kernel_size=1)

        if checkpoint_path:
            self.load_pretrained(checkpoint_path)
        


    # -------- 位置编码 / backbone 输出整理 --------

    def forward(self, vis_image, inf_image, gt=None):
            batch_size = vis_image.shape[0]
            # print(batch_size)
            fused_bb, rgb_bb, t_bb = self.forward_image(vis_image, inf_image)
            masks1 = self._decode_from_backbone(fused_bb, batch_size)
            masks2 = self._decode_from_backbone(rgb_bb, batch_size)
            masks3 = self._decode_from_backbone(t_bb, batch_size)
            masks = masks1 + masks2 + masks3
            if self.training and gt is not None:
            # 在各张卡上独立计算 Loss
            # 注意：这里调用的 iou_loss 和 dice_loss 必须是无状态的函数
                loss_iou_1 = iou_loss(masks1, gt)
                loss_dice_1 = dice_loss(masks1, gt)
                loss_iou_2 = iou_loss(masks2, gt)
                loss_dice_2 = dice_loss(masks2, gt)
                loss_iou_3 = iou_loss(masks3, gt)
                loss_dice_3 = dice_loss(masks3, gt)
                loss_iou_4 = iou_loss(masks, gt)
                loss_dice_4 = dice_loss(masks, gt)

                loss = (loss_iou_1 + loss_dice_1) + (loss_iou_2 + loss_dice_2) + (loss_iou_3 + loss_dice_3) + (loss_iou_4 + loss_dice_4)
                # 返回 Loss 而不是预测图
                return loss.unsqueeze(0)
            else:
                # 测试模式或不传 gt 时，返回预测图
                return masks1, masks2, masks3, masks
    def forward_image(self, img_batch: torch.Tensor, t_batch: torch.Tensor):
        # print(img_batch.shape, t_batch.shape)
        state_rgb = {}
        state_t = {}
        state_fuse = {}
        state_rgb ["original_heights"] = [image.shape[1] for image in img_batch]
        state_rgb ["original_widths"] = [image.shape[2] for image in img_batch]
        state_t ["original_heights"] = [image.shape[1] for image in t_batch]
        state_t ["original_widths"] = [image.shape[2] for image in t_batch]
        state_fuse ["original_heights"] = [image.shape[1] for image in img_batch]
        state_fuse ["original_widths"] = [image.shape[2] for image in img_batch]
        rgb_feature = self.patch_embed(img_batch)
        t_feature = self.patch_embed(t_batch)
        h, w = rgb_feature.shape[1], rgb_feature.shape[2]
        if self.pos_embed is not None:
            rgb_feature = rgb_feature + get_abs_pos(
                self.pos_embed,
                True,
                (h, w),
                False,
                tiling=True,
            )
            t_feature = t_feature + get_abs_pos(
                self.pos_embed,
                True,
                (h, w),
                False,
                tiling=True,
            )
        rgb_feature = self.ln_pre(rgb_feature)
        t_feature = self.ln_pre(t_feature)
        # with torch.no_grad():
        #     r_cpu = rgb_feature.detach().cpu()
        #     t_cpu = t_feature.detach().cpu()
        #     print(r_cpu.min().item(), r_cpu.max().item(),t_cpu.min().item(), t_cpu.max().item())

        # rgb_feature = rgb_feature + self._get_pos_embed(rgb_feature.shape[1:3])
        # t_feature = t_feature + self._get_pos_embed(t_feature.shape[1:3])

        rgb_outputs, t_outputs = [], []

        for i, block in enumerate(self.blocks):
            rgb_feature, t_feature = block(rgb_feature, t_feature)
            # with torch.no_grad():
            #     r_cpu = rgb_feature.detach().cpu()
            #     t_cpu = t_feature.detach().cpu()
            #     print(r_cpu.min().item(), r_cpu.max().item(),t_cpu.min().item(), t_cpu.max().item())
            rgb_feature = rgb_feature + self.rgb_adapters[i](rgb_feature)
            t_feature = t_feature + self.t_adapters[i](t_feature)
            # if (i == self.full_attn_ids[-1]) or (
            #     self.return_interm_layers and i in self.full_attn_ids
            # ):
            if i == self.full_attn_ids[-1]:
                rgb_feat = self.ln_post(rgb_feature)
                t_feat = self.ln_post(t_feature)
                rgb_feats = rgb_feat[:, 0:]
                t_feats = t_feat[:, 0:]
                if rgb_feats.ndim == 4:
                    rgb_feats = rgb_feats.permute(0, 3, 1, 2)
                    t_feats = t_feats.permute(0, 3, 1, 2)
                    rgb_outputs.append(rgb_feats)
                    t_outputs.append(t_feats)
        fused_outputs = []
        fused = torch.cat([rgb_outputs[0], t_outputs[0]], dim=1)  # [B, 2C, H, W]
        fused = self.fuse_convs(fused)  # [B, C, H, W]
        fused_outputs.append(fused)
        state_fuse ["backbone_out"] = self.tobackboneout(fused_outputs)
        state_rgb ["backbone_out"] = self.tobackboneout(rgb_outputs)
        state_t ["backbone_out"] = self.tobackboneout(t_outputs)
        return state_fuse, state_rgb, state_t

        # return fused_backbone, rgb_backbone, t_backbone
    
    def tobackboneout(self, xs):
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = [], []
        x = xs[-1]  # simpleFPN
        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)
            sam2_x_out = self.sam2_convs[i](x)
            sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
            sam2_out.append(sam2_x_out)
            sam2_pos.append(sam2_pos_out)
        # return sam3_out, sam3_pos, sam2_out, sam2_pos
        sam3_out, sam3_pos = (
                sam3_out[: -1],
                sam3_pos[: -1],
            )
        sam2_out, sam2_pos = (
                sam2_out[: -1],
                sam2_pos[: -1],
            )
        sam2_output = None
        sam2_src = sam2_out[-1]
        sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_out,
            }
        sam3_src = sam3_out[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_out,
            "sam2_backbone_out": sam2_output,
        }
        # t = output["sam2_backbone_out"]["backbone_fpn"][0]
        # with torch.no_grad():
        #     t_cpu = t.detach().cpu()
        #     print(t_cpu.min().item(), t_cpu.max().item())
        output["sam2_backbone_out"]["backbone_fpn"][0] = (
            self.mask_decoder.conv_s0(
                output["sam2_backbone_out"]["backbone_fpn"][0]
            )
        )
        output["sam2_backbone_out"]["backbone_fpn"][1] = (
            self.mask_decoder.conv_s1(
                output["sam2_backbone_out"]["backbone_fpn"][1]
            )
        )
        return output

    # -------- 主干前向：RGB / T + KAN + GatedAdapter --------



    # -------- SAM 解码部分保留不变 --------

    def _decode_from_backbone(self, inference_state, batch_size):
        backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
        feature_maps = backbone_out["backbone_fpn"][-3 :]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        # Add no_mem_embed, which is added to the lowest res feat. map during training on videos
        # vision_feats[-1] = (
        #     vision_feats[-1] + self.no_mem_embed
        # )
        batch_size = vision_feats[-1].shape[1]
        orig_heights, orig_widths = (
            inference_state["original_heights"],
            inference_state["original_widths"],
        )
        assert (
            batch_size == len(orig_heights) == len(orig_widths)
        ), f"Batch size mismatch in predict_inst_batch. Got {batch_size}, {len(orig_heights)}, {len(orig_widths)}"
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self._bb_feat_sizes[::-1]
            )
        ][::-1]
        features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        num_images = len(features["image_embed"])
        all_masks = []
        concat_points, boxes, mask_input = None, None, None
        for img_idx in range(num_images):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            high_res_features = [
                feat_level[img_idx].unsqueeze(0)
                for feat_level in features["high_res_feats"]
            ]
            low_res_masks, _, _, _ = self.mask_decoder(
                image_embeddings=features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            masks = F.interpolate(
                low_res_masks.float(),
                (504, 504),
                mode="bilinear",
                align_corners=False,
            )
            all_masks.append(masks)
        return torch.cat(all_masks, dim=0)


    def load_pretrained(self, pretrained_path, print_loaded=True, max_print=200):
        print(f"Loading pretrained model from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # 兼容常见保存格式：直接是 state_dict 或 {'state_dict': ...}
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 去除可能存在的 "module." 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        # 先拿当前模型的 key，用于判断哪些“真正匹配并会加载”
        model_state = self.state_dict()
        loaded_keys = []
        shape_mismatch = []

        for k, v in new_state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    loaded_keys.append(k)
                else:
                    shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        # ===== 打印统计 =====
        # print(f"[Pretrained] Total ckpt tensors: {len(new_state_dict)}")
        # print(f"[Pretrained] Loaded (matched key & shape): {len(loaded_keys)}")
        # print(f"[Pretrained] Missing keys in ckpt (model expects): {len(missing)}")
        # print(f"[Pretrained] Unexpected keys in ckpt: {len(unexpected)}")
        # print(f"[Pretrained] Shape mismatches: {len(shape_mismatch)}")
