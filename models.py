
import os
import timm
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models
from torchvision import transforms as T
from .tinc.main import get_vicreg_model
from .vit_encoder import SpatioTemporalEncoder
from .RETFound.util.pos_embed import interpolate_pos_embed
from  .RETFound.models_vit import RETFound_mae
from timm.models.layers import trunc_normal_
import inspect
import numpy as np
import pytorch_lightning as pl
print('model params',inspect.signature(RETFound_mae))

#-----------------------Models----------------------#
def retfound_encoder(img_size=224, root = ''):
    model =RETFound_mae(
    img_size = img_size,
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=False,    
)
   
    # weights downloaded from https://drive.google.com/uc?id=1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE on 22 May, 2025 from https://huggingface.co/open-eye/RETFound_MAE
    # load RETFound weights
    checkpoint = torch.load(f'{root}/retfound_cfp.pt', map_location='cpu', weights_only=False) #the weight only has the encoder
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    
    
    if 'head.weight' in checkpoint_model:
        num_output_nodes = checkpoint_model['head.weight'].shape[0]
        print(f"----------- \n \n Number of output nodes in the pretrained model: {num_output_nodes} ----------- \n \n ")
    else:
        print("----------- \n \n  head.weight not found in checkpoint. Cannot determine output size. ----------- \n \n ")

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # # load pre-trained model
    model.load_state_dict(checkpoint_model, strict=False)

    # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    # # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    model.head = nn.Identity()
    embed_dim = model.head.in_features if isinstance(model.head, nn.Linear) else model.embed_dim
    return model, embed_dim


def load_dinov2_weights(model, ckpt_path):

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if model.chunked_blocks:
    # Flatten chunked model: compute mapping from flat index → chunk/block indices
        print('flattening chunk')
        new_state = {}

        for k, v in checkpoint.items():
            if not k.startswith("blocks."):
                new_state[k] = v

        flat_idx = 0
        for chunk_idx, block_chunk in enumerate(model.blocks):
            n_blocks = len(block_chunk.blocks) if hasattr(block_chunk, "blocks") else len(block_chunk)
            for block_idx in range(n_blocks):
                for k, v in checkpoint.items():
                    if k.startswith(f"blocks.{flat_idx}."):
                        suffix = k.split(f"blocks.{flat_idx}.")[-1]
                        new_key = f"blocks.{chunk_idx}.{block_idx}.{suffix}"
                        new_state[new_key] = v
                flat_idx += 1
        state_to_load = new_state
    else:
        # flat model, just load with strict=False
        state_to_load = checkpoint

    if "mask_token" not in state_to_load:
        nn.init.normal_(model.mask_token, std=1e-6)
        state_to_load["mask_token"] = model.mask_token.detach()

    if "pos_embed" in state_to_load:
        checkpoint_pos_embed = state_to_load["pos_embed"]
        model_num_patches = model.patch_embed.num_patches
        checkpoint_num_patches = checkpoint_pos_embed.shape[1] - 1
        
        if checkpoint_num_patches != model_num_patches:
            print(f'Interpolating pos_embed from {checkpoint_num_patches} to {model_num_patches} patches')
            
            # Save current pos_embed
            original_pos_embed = model.pos_embed
            
            # Temporarily set to checkpoint pos_embed
            model.pos_embed = nn.Parameter(checkpoint_pos_embed)
            
            # Calculate target image size
            h = w = int(model_num_patches ** 0.5) * model.patch_size
            
            # Create dummy input
            dummy_x = torch.zeros(1, model_num_patches + 1, model.embed_dim)
            
            # Use model's interpolation method
            interpolated_pos_embed = model.interpolate_pos_encoding(dummy_x, w, h)
            
            # Restore original pos_embed
            model.pos_embed = original_pos_embed
            
            # Update state with interpolated version
            state_to_load["pos_embed"] = interpolated_pos_embed


    missing_keys, unexpected_keys = model.load_state_dict(state_to_load, strict=True)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    return model, model.embed_dim

def load_dino_nako(weights_path = None, config_file = None, nako_model = True):
    from .dino import build_model_from_cfg
    from omegaconf import OmegaConf


    cfg = OmegaConf.load(config_file)
    _, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
    
    checkpoints = torch.load(weights_path, map_location= 'cpu')
    print("\n \n ***dino model checkpointkeys*** \n \n",checkpoints.keys())

    if nako_model:
        print('loading nako dino model')
        backbone_state_dict = {k.replace('backbone.', ''):v
                            for k, v in  checkpoints['teacher'].items() 
                            if k.startswith('backbone.')}
        teacher_backbone.load_state_dict(backbone_state_dict, strict=True)
    else:
        print('loading dino pretrained model')
        teacher_backbone.load_state_dict(checkpoints, strict=True)
    return teacher_backbone, embed_dim

class DINOBackbone(nn.Module):
    """Feature extractor from DINOv2 encoder."""
    
    def __init__(self, model: nn.Module, use_cls: bool = True,
                  use_registers: bool = False):
        super().__init__()
        self.model = model
        self.use_cls = use_cls
        self.use_registers = use_registers
        
        # Get feature dimension from the model
        # DINOv2 models have embed_dim attribute
        if hasattr(model, 'embed_dim'):
            self.feat_dim = model.embed_dim
            print('has embed_dim with shape', self.feat_dim)
        elif hasattr(model, 'num_features'):
            self.feat_dim = model.num_features
            print('has num_features with shape', self.feat_dim)
        else:
            # Fallback: infer from cls_token or pos_embed
            if hasattr(model, 'cls_token'):
                print('has cls_token')
                self.feat_dim = model.cls_token.shape[-1]
            elif hasattr(model, 'pos_embed'):
                print('has pos_embed')
                self.feat_dim = model.pos_embed.shape[-1]
            else:
                raise ValueError("Cannot determine feature dimension")
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            features: (B, feat_dim) feature vectors
        """
        # DINOv2 backbone returns a dictionary with keys:
        # - 'x_norm_clstoken': (B, D) - normalized CLS token
        # - 'x_norm_regtokens': (B, n_reg, D) - normalized register tokens
        # - 'x_norm_patchtokens': (B, N, D) - normalized patch tokens
        # - 'x_prenorm': (B, 1+n_reg+N, D) - pre-normalization features
        # - 'masks': mask info if any
        # is_training=False means no masking, straightforward inference
        output_dict = self.model(x, is_training=True) #is_training just returns all features for loss computation and has nothing to do with eval or train mode
        
        if self.use_cls:
            # Use CLS token (recommended for DINOv2)
            features = output_dict['x_norm_clstoken']  # (B, D)
        else:
            # Use mean of patch tokens
            features = output_dict['x_norm_patchtokens'].mean(dim=1)  # (B, D)
        
        return features
class ViTBackbone(nn.Module):
    """Feature extractor from  encoder."""
    def __init__(self, model: nn.Module, use_cls: bool = True):
        super().__init__()
        self.model = model
        self.use_cls = use_cls

        self.feat_dim = model.cls_token.shape[-1]

    def forward(self, x):
        # no masking for downstream
        latent = self.model.forward_features(x)  # (B, 1+N, D)
        # print('\n \n ***latent shape in vitbackbone*** \n \n', latent.keys())

        if self.use_cls:
            return latent[:, 0]              # (B, D) CLS token
        else:
            return latent[:, 1:, :].mean(1)   # (B, D) mean over patch tokens


def default(val, def_val):
    return def_val if val is None else val

def strip_prefix_if_present(state_dict, prefix):
    if any(k.startswith(prefix) for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict
    return state_dict

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, sync_batchnorm=None):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, projection_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    

def load_lightning_model(weights_path, lightning_model, mae):
    model_light_chkpt = torch.load(weights_path, map_location='cpu', weights_only = False)
    print(lightning_model)
    lit = lightning_model(mae)
    lit.load_state_dict(model_light_chkpt["state_dict"], strict=True)
    print('Loaded model')
    lit = lit.to('cuda')
    return lit
 

class SwinEncoder(nn.Module):
    def __init__(self, swin_mae):
        super().__init__()
        # copy only encoder parts
        self.patch_embed = swin_mae.patch_embed
        self.layers = swin_mae.layers

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)  # each layer = BasicBlock
        return x  # features from the last BasicBlock (before decoder)

def mae_nako_weights(model_weight):
    from  .RETFound import models_mae
    from .lightning_helpers import OptimCfg, MAELightning

    # model_weight = '/gpfs01/berens/user/inwabufo/survcnn_skeleton/survival_on_embedding/output/mae_weights_only.pt'

    mae = models_mae.mae_vit_base_patch16_dec512d8b(norm_pix_loss=True)
    lit = MAELightning(mae) # this is called because I want the exact structure used for training but it is the model_mae forward function that is executed
    model_light_chkpt = torch.load(model_weight, map_location='cpu', weights_only = True)
    lit.load_state_dict(model_light_chkpt["state_dict"], strict=True)
    print('Loaded model')
    lit = lit.to('cuda')
    model = lit.mae
    embed_dim = model.patch_embed.proj.out_channels
    return model, embed_dim
    
class MAEBackbone(nn.Module):
    """Feature extractor from  MaskedAutoencoderViT encoder."""
    def __init__(self, mae: nn.Module, use_cls: bool = True):
        super().__init__()
        self.mae = mae
        self.use_cls = use_cls

        self.feat_dim = mae.cls_token.shape[-1]

    def forward(self, x):
        # no masking for downstream
        latent, _, _ = self.mae.forward_encoder(x, mask_ratio=0.0)  # (B, 1+N, D)
        # print(f"latent shape: {latent.shape}, use_cls: {self.use_cls}")  # always prints

        if self.use_cls:
            return latent[:, 0]               # (B, D) CLS token
        else:
            return latent[:, 1:, :].mean(1)   # (B, D) mean over patch tokens

class ModelWithMAE(nn.Module):
    def __init__(self, mae, use_cls=True):
        super().__init__()
    
        self.backbone = MAEBackbone(mae, use_cls=use_cls)
        self.head =  nn.Identity()

    def forward(self, x):
        feats = self.backbone(x)   # (B, D)
        f_feats = self.head(feats)
        return f_feats
    

def load_dino_pretrained(eval_augur = False):
    if eval_augur:
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', 
                          pretrained=False,
                          img_size = 518)
    else:
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', 
                          pretrained=True,
                          img_size = 518)

    
    return model

    
class SimCLR(nn.Module):
    def __init__(self, backbone, in_dim=512, 
                 hidden_dim=1024, feat_dim=128,  **kwargs):
        
        super().__init__()

        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.projection_head = MLP(self.in_dim, 
                                   self.feat_dim, self.hidden_dim)

    def forward(self, x):
        h = self.backbone(x)
        h = torch.flatten(h, 1) 
        z = self.projection_head(h)
        return h, z 


def resnet18_modified():
    resnet18_model = models.resnet18(weights = None)
    resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet18_model.fc = nn.Identity()
    return resnet18_model

def resnet18_modified_pretrained():
    resnet18_model = models.resnet18(weights='IMAGENET1K_V1')
    resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet18_model.fc = nn.Identity()
    return resnet18_model


RESNET_MODEL_DICT = {
                    "resnet18": resnet18_modified_pretrained(),
                     "resnet18_modified": resnet18_modified(),
                     "resnet18_modified_pretrained": resnet18_modified_pretrained()
                     

}
