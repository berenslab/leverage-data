# from https://github.com/RobvanGastel/dinov3-finetune/blob/main/dino_finetune/model/dino.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_lora_to_dino_student(student: nn.ModuleDict, r: int, alpha: int = 1):
    backbone = student["backbone"]

    # freeze everything 
    for p in backbone.parameters():
        p.requires_grad = False

    bad = []
    for name, p in backbone.named_parameters():
        if p.requires_grad and ('linear_a' not in name and 'linear_b' not in name):
            bad.append(name)
        if bad:
            raise ValueError(f"WARNING: {name} is trainable but shouldn't be!")  

    print("=== After freeze ===")
    trainable_count = sum(p.requires_grad for p in backbone.parameters())
    print(f"Trainable params in backbone: {trainable_count}")  
    trainable_scalars = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print("Trainable scalars:", trainable_scalars)


    for block in backbone.blocks:
        if isinstance(block.attn.qkv, LoRA):
            continue

        qkv = block.attn.qkv
        dim = qkv.in_features

        # q adapter
        a_q = nn.Linear(dim, r, bias=False) # A
        b_q = nn.Linear(r, dim, bias=False) # B

        # v adapter
        a_v = nn.Linear(dim, r, bias=False) # A 
        b_v = nn.Linear(r, dim, bias=False) # B

        # init like LoRA
        nn.init.kaiming_uniform_(a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(b_q.weight)
        nn.init.kaiming_uniform_(a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(b_v.weight)

        # use lora in the attention block
        block.attn.qkv = LoRA(qkv, a_q, b_q, a_v, b_v, alpha=alpha)
    print("=== After Lora ===")
    trainable_count_lora = sum(p.requires_grad for p in backbone.parameters())
    print(f"Trainable params in backbone after lora: {trainable_count_lora}")  
    trainable_scalars_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print("Trainable scalars lora:", trainable_scalars_lora)
    student['backbone'] = backbone
    return student

class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        alpha: int = 1,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        # self.w_identity = torch.eye(self.dim)

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.scaling = alpha / linear_a_q.out_features

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x)) * self.scaling
        new_v = self.linear_b_v(self.linear_a_v(x)) * self.scaling

        # Add new q and v components to the original qkv tensor
        qkv = qkv.clone()
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v

        return qkv


# class DINOEncoderLoRA(nn.Module):
#     def __init__(
#         self,
#         encoder,
#         r: int = 3,
#         # emb_dim: int = 1024,
#         patch_size: int = 14,
#         img_dim: tuple[int, int] = (520, 520),
#     ):
#         """The DINOv2 encoder model for finetuning to downstream tasks.

#         Args:
#             encoder (nn.Module): The ViT encoder model loaded with the DINOv2 model weights.
#             r (int, optional): The rank parameter of the LoRA weights. Defaults to 3.
#             emb_dim (int, optional): The embedding dimension of the encoder. Defaults to 1024.
#             n_classes (int, optional): The number of classes to output. Defaults to 1000.
#             img_dim (tuple[int, int], optional): The input image dimension. Defaults to
#                 (520, 520).
#         """
#         super().__init__()
#         assert img_dim[0] % patch_size == 0, "Wrong input shape for patches"
#         assert r > 0

#         # self.emb_dim = emb_dim
#         self.img_dim = img_dim
#         self.encoder = encoder # the entire student model including the backbone, dino head and ibot head
#         self.backbone = encoder['backbone']
        

#         for param in self.encoder.parameters(): # freeze the entire model 
#             param.requires_grad = False

#         # Add LoRA layers to the encoder
#         self.lora_layers = list(range(len(self.backbone.blocks)))
#         self.w_a = []
#         self.w_b = []

#         for i, block in enumerate(self.backbone.blocks):
#             if i not in self.lora_layers:
#                 continue

#             # prevent wrapping twice
#             if isinstance(block.attn.qkv, LoRA):
#                 continue

#             w_qkv_linear = block.attn.qkv
#             dim = w_qkv_linear.in_features

#             w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
#             w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

#             self.w_a.extend([w_a_linear_q, w_a_linear_v])
#             self.w_b.extend([w_b_linear_q, w_b_linear_v])

#             block.attn.qkv = LoRA(
#                 w_qkv_linear,
#                 w_a_linear_q,
#                 w_b_linear_q,
#                 w_a_linear_v,
#                 w_b_linear_v,
#             )
#         self._reset_lora_parameters()

#     def _create_lora_layer(self, dim: int, r: int):
#         w_a = nn.Linear(dim, r, bias=False)
#         w_b = nn.Linear(r, dim, bias=False)
#         return w_a, w_b

#     def _reset_lora_parameters(self) -> None:
#         for w_a in self.w_a:
#             nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
#         for w_b in self.w_b:
#             nn.init.zeros_(w_b.weight)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         feature = self.backbone.forward_features(x)
#         # get the patch embeddings - so we exclude the CLS token
#         patch_embeddings = feature["x_norm_patchtokens"]
#         return feature, patch_embeddings

#     def save_parameters(self, filename: str) -> None:
#         """Save the LoRA weights and decoder weights to a .pt file

#         Args:
#             filename (str): Filename of the weights
#         """
#         w_a, w_b = {}, {}
       
#         w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
#         w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}

#         torch.save({**w_a, **w_b}, filename)

#     def load_parameters(self, filename: str) -> None:
#         """Load the LoRA and decoder weights from a file

#         Args:
#             filename (str): File name of the weights
#         """
#         state_dict = torch.load(filename)

#         for i, w_A_linear in enumerate(self.w_a):
#             saved_key = f"w_a_{i:03d}"
#             saved_tensor = state_dict[saved_key]
#             w_A_linear.weight = nn.Parameter(saved_tensor)

#         for i, w_B_linear in enumerate(self.w_b):
#             saved_key = f"w_b_{i:03d}"
#             saved_tensor = state_dict[saved_key]
#             w_B_linear.weight = nn.Parameter(saved_tensor)

        