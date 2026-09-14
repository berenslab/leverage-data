import math
import torch
from torch import nn
import torch.nn.functional as F
# from models.vit_helper import ViT_Backbone
# from models.videovit import TimeEmbedding
from timm.models.layers import trunc_normal_, to_2tuple
from einops.layers.torch import Rearrange


class ViT_Backbone(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, attn_mask=None):

        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [B, N]
                attn_mask = attn_mask[:, None, :]  # [B, 1, N]
                attn_mask = attn_mask & attn_mask.transpose(-1, -2)  # [B, N, N] pairwise AND
                attn_mask = attn_mask[:, None, :, :]  # → [B, 1, N, N]

            elif attn_mask.dim() == 3:  # [B, N, N]
                attn_mask = attn_mask[:, None, :, :]     # → [B, 1, N, N]

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return x


    def get_last_selfattention(self, x, attn_mask=None):
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, attn_mask=attn_mask)
            else:
                return blk(x, return_attention=True, attn_mask=attn_mask)

    def get_intermediate_layers(self, x, n=1, attn_mask=None):
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        if attn_mask is not None:
            # Debug info
            # print(f"[DEBUG] attn.shape: {attn.shape}, attn_mask.shape: {attn_mask.shape}")
            assert attn.shape[-1] == attn_mask.shape[-1], "Mismatch in key dimension"
            assert attn.shape[-2] == attn_mask.shape[-2], "Mismatch in query dimension"

            
            # Reshape mask to match attention dimensions
            if attn_mask.dim() == 2:  # [B, N]
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # → [B, 1, 1, N]
                # Broadcast to match N dimension
                attn_mask = attn_mask.expand(-1, -1, attn.size(-2), -1)
            elif attn_mask.dim() == 3:  # [B, N, N]
                attn_mask = attn_mask.unsqueeze(1)  # → [B, 1, N, N]
            
            # Verify shapes are compatible for masked_fill
            if attn.shape[-1] != attn_mask.shape[-1] or attn.shape[-2] != attn_mask.shape[-2]:
                raise ValueError(
                    f"Attention and mask shapes incompatible: attn={attn.shape}, mask={attn_mask.shape}. "
                    f"Expected last two dimensions to match."
                )
            assert attn.shape[-1] == attn_mask.shape[-1], "Mask last dimension must match attention"
            assert attn.shape[-2] == attn_mask.shape[-2], "Mask second-last dimension must match attention"

            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, attn_mask=None):
        y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return attn, x
        return x
class SimplePatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=3, embed_dim=384):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # print(f"[*] Image size: {img_size}, Patch size: {patch_size}")

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            "Image dimensions must be divisible by patch size"

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(patch_size[0] * patch_size[1] * in_chans, embed_dim)
        )

    def forward(self, x):
        return self.proj(x)
def return_vit_config(vit_config):
    if vit_config == "tiny":
        return 192, 12, 3
    elif vit_config == "small":
        return 384, 12, 6
    elif vit_config == "small_st":
        return 384, 15, 6
    elif vit_config == "base":
        return 768, 12, 12
    elif vit_config == "large":
        return 1024, 24, 16
    else:
        raise NotImplementedError(f"vit_config: {vit_config} is not available")
class VideoViT(nn.Module):
    def __init__(self, image_only=True, frame_size=28, channels=3, patch_spatial=7, 
                 vit_config="small"):
        super().__init__()
        self.image_only = image_only
        _, depth, num_heads = return_vit_config(vit_config)
        embed_dim = 384
        self.embed_dim = embed_dim

        

        self.patch_embed = SimplePatchEmbed(
            img_size=frame_size,
            patch_size=patch_spatial,
            in_chans=channels,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(self, x):
        B, N, D = x.shape
        N_input = N - 1  # exclude cls token
        N_orig = self.pos_embed.shape[1] - 1

        if N_input == N_orig:
            return self.pos_embed

        cls_token = self.pos_embed[:, :1, :]
        patch_embed = self.pos_embed[:, 1:, :]
        size_orig = int(math.sqrt(N_orig))
        size_new = int(math.sqrt(N_input))
        patch_embed = patch_embed.reshape(1, size_orig, size_orig, D).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(size_new, size_new), mode='bilinear', align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, -1, D)

        return torch.cat((cls_token, patch_embed), dim=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2)  # add dummy time dimension

        B, C, T, H, W = x.shape
        assert T == 1, "VideoViT in image-only mode expects T=1"
        x = x.squeeze(2)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.interpolate_pos_encoding(x)
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # return only cls token


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, 
                 img_size= None,
                 patch_spatial= None, 
                 arch= None,
                 temporal_dim= None,
                 temporal_depth= None,
                 use_time_embed= None,
                 learnable_time_embed= None,
                 clip_frames= None):
        super(SpatioTemporalEncoder, self).__init__()


        self.spatio_encoder = VideoViT(
            frame_size=img_size,
            channels=3,
            patch_spatial = patch_spatial,
            vit_config=arch,
        )

        dummy_input = torch.randn(2, 3, 1, img_size, img_size)  # [B, C, T=1, H, W]
        true_output = self.spatio_encoder(dummy_input)
        # print(f"[INFO] VideoViT output shape: {true_output.shape}")

        self.embed_dim = true_output.shape[-1]  # dynamically detect output dim

        if temporal_dim != self.embed_dim:
            self.temporal_dim = self.embed_dim
        else:
            self.temporal_dim = temporal_dim

        self.mid_project = nn.Identity()
        self.temporal_depth =  temporal_depth #cfg.get("temporal_depth", 3)

        # ---- Tokens ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_dim))
        self.downstream_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.use_time_embed = use_time_embed#cfg.use_time_embed
        # print(f"[*] Using time embedding: {self.use_time_embed}")

        if self.use_time_embed:
            print(' ------ \n \n you are using time embeddings earnable time embeddings, learnable_time_embed \n \n --------')
            self.time_embed = TimeEmbedding(self.temporal_dim, learnable=learnable_time_embed)
            print(f' ------ temporal_dim shape {self.temporal_dim} time_embed shape  --------')
            self.pos_embed = nn.Parameter(torch.zeros(1, clip_frames, self.embed_dim))
        else:
            print(' ------ \n \n you are not using time embeddings \n \n --------')

            self.pos_embed = nn.Parameter(torch.zeros(1, clip_frames, self.embed_dim))

        # ---- Temporal encoder ----
        self.temporal_encoder = self.temporal_encoder = ViT_Backbone(
                embed_dim=self.embed_dim,   # Use embed_dim from the spatial encoder, always
                depth=self.temporal_depth,
                num_heads=6,
                mlp_ratio=4
)
       

        # ---- Initialize ----
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.downstream_cls_token, std=0.02)
        self.apply(self._init_weights)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {"downstream_cls_token", "cls_token"}

    def _init_weights(self, m):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.downstream_cls_token, std=0.02)
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_spatio_encoder(self, x):
        if isinstance(x, list):
            x = torch.tensor(torch.cat(x))
            print(type(x), len(x), type(x[0]), x[0].shape, x[1].shape)


        if x.ndim == 4:
            # Single frame case: [B, C, H, W]
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        b, t, c, h, w = x.shape  # Batch, Time, Channel, Height, Width

        # Reshape to process each frame individually:
        x = x.view(b * t, c, h, w)  # [B*T, C, H, W]

        # VideoViT expects [B, C, T=1, H, W]
        x = x.unsqueeze(2)  # [B*T, C, 1, H, W]

        # print(f"[DEBUG] Passing {x.shape} into spatial encoder...")
        x = self.spatio_encoder(x)  # Should output [B*T, embed_dim]

        # print(f"[DEBUG] spatio encoder output shape: {x.shape}, expected embed_dim: {self.embed_dim}")

        if x.shape[-1] != self.embed_dim:
            print("[WARNING] Spatial encoder output dim does not match expected embed_dim.")
            print(f"Got {x.shape[-1]} vs expected {self.embed_dim}")

        # Reshape back to [B, T, embed_dim]
        x = x.view(b, t, -1)

        return x


    def prepare_temporal_token(self, x, timestep):
        B, T, D = x.shape  # B = batch, T = time (sequence), D = dim

        # Adjust pos_embed to match current time tokens T
        if self.pos_embed.shape[1] != T:
            # print(f"[DEBUG] Adjusting pos_embed from {self.pos_embed.shape[1]} to {T}")
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 2, 1),  # (1, D, T_base)
                size=T,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # back to (1, T, D)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed  # broadcast-safe

        if self.use_time_embed:
            x = x + self.time_embed(timestep)

        return x


    def forward_temporal_encoder(self, x, attn_mask, time_step, prepare_token=True):
        if prepare_token:
            x = self.prepare_temporal_token(x, time_step)

        # Add the [CLS] token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Adjust the attn_mask to account for the CLS token
        if attn_mask is not None:
            cls_mask = torch.ones((attn_mask.shape[0], 1), dtype=torch.bool, device=attn_mask.device)
            attn_mask = torch.cat((cls_mask, attn_mask), dim=1)  # Now [B, T+1]

        x = self.temporal_encoder(x, attn_mask=attn_mask)
        return x

    def forward(self, x, attn_mask, time_step, feat_op="cls", return_spatial=False):
        # print(f"[DEBUG] Final encoder output shape before head: {x.shape}")

        x_spatial = self.forward_spatio_encoder(x)

        # 💥 Don't use spatial attn_mask here
        B, T, D = x_spatial.shape
        temporal_attn_mask = torch.ones((B, T), dtype=torch.bool, device="cuda")  # All visible by default

        x = self.forward_temporal_encoder(x_spatial, temporal_attn_mask, time_step)


  
        if feat_op == "cls":
            x = x[:, 0, :]
        elif feat_op == "pool":
            # remove the cls token
            x = x[:, 1:, :]
            # average pool the embeddings where attention mask is 1
            x *= attn_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attn_mask.sum(dim=1, keepdim=True)
        if return_spatial:
            return x_spatial, x
        return x
