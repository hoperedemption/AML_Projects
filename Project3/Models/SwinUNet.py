import torch 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
from math import sqrt, log
from einops import rearrange
from itertools import product
from timm.layers import DropPath

# patch partition + linear embedding
class PatchEmbed(nn.Module):
    def __init__(self, in_dim, img_h, img_w, patch_h, patch_w, embed_dim):
        super().__init__()
        
        self.in_dim = in_dim
        self.img_h = img_h
        self.img_w = img_w
        self.patch_h = patch_h 
        self.patch_w = patch_w
        self.embed_dim = embed_dim
        
        self.nb_h_patches = (img_h // patch_h)
        self.nb_w_patches = (img_w // patch_w)
        self.n_patches = self.nb_h_patches * self.nb_w_patches
        self.proj = nn.Conv2d(in_dim, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed the patches

        Args:
            x (torch.Tensor): image input of shape (B, C, H, W)

        Returns:
            torch.Tensor: (B, n_patches, embed_dim)
        """
        x = self.proj(x) # (B, embed_dim, nb_h_patches, nb_w_patches)
        x = rearrange(x, 'B E nbH nbW -> B (nbH nbW) E', nbH=self.nb_h_patches, nbW=self.nb_w_patches) # (B, n_patches, embed_dim)
        return x  

# mlp for transformer block
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, dropout=.0):
        super().__init__()
        self.in_dim = in_dim 
        self.hid_dim = hid_dim or in_dim 
        self.out_dim = out_dim or in_dim
        
        self.arch = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim), 
            nn.GELU(approximate='tanh'), 
            nn.Linear(self.hid_dim, self.out_dim), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.arch(x)

# window msa
class WindowMSA(nn.Module):
    def __init__(self, in_dim, wdw_size, scale=8, n_heads=1, bias=True, att_drop=.0, proj_drop=.0):
        super().__init__()
        self.in_dim = in_dim 
        self.wdw_h, self.wdw_w = wdw_size
        self.n_heads = n_heads 
        self.bias = bias 
        self.att_drop = att_drop
        self.proj_drop = proj_drop
        self.scale = scale
        
        self.key_dim = 3 * self.in_dim 
        self.val_dim = 3 * self.in_dim
        
        # compute the relative positions
        relative_pos = self._pos(self.scale)
        self.register_buffer('relative_pos', relative_pos)
        
        # compute the relative indices
        relative_idx = self._index()
        self.register_buffer('relative_idx', relative_idx)
        
        # mlp for the continuous position bias
        self.mlp_bias = nn.Sequential(
            nn.Linear(2, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.n_heads, bias=False)
        )

        # the embedding projections
        self.WQ = nn.Linear(self.in_dim, self.key_dim * self.n_heads, bias=bias)
        self.WK = nn.Linear(self.in_dim, self.key_dim * self.n_heads, bias=False)
        self.WV = nn.Linear(self.in_dim, self.val_dim * self.n_heads, bias=bias)
        
        # the out projection
        self.WA = nn.Linear(self.val_dim * self.n_heads, self.in_dim)
        
        # dropouts
        self.attention_drop = nn.Dropout(self.att_drop)
        self.projection_drop = nn.Dropout(self.proj_drop)
        
        # logits scale
        self.logit_scale = nn.Parameter(torch.log(10 * (torch.rand((self.n_heads, 1, 1), dtype=torch.float32) + 1)), requires_grad=True)
        self.scale_const = log(1e2)
        
    def forward(self, E, mask=None):
        # (B*n_patches, N=num_wdw, C) -> (B*n_patches, N=num_wdw, n_heads * d)
        Q, K, V = self.WQ(E), self.WK(E), self.WV(E) 
        
        # (B*n_patches, N, n_heads * d) -> (B*n_patches, n_heads, N, d)
        swap_dim = lambda mat, dim: rearrange(mat, 'B N (H D) -> B H N D', H=self.n_heads, D=dim)
        Q, K, V = swap_dim(Q, self.key_dim), swap_dim(K, self.key_dim), swap_dim(V, self.val_dim)
        
        # compute cosine similarity attention
        Q, K = F.normalize(Q, dim=-1), F.normalize(K, dim=-1)
        
        # (B*n_patches, n_heads, N, d) @ (B*n_patches, n_heads, d, N) -> (B*n_patches, n_heads, N, N)
        P = torch.einsum('bhnd,bhmd->bhnm', Q, K)
        scaled_logits = self.logit_scale.clamp(max=self.scale_const).exp()
        P *= scaled_logits
        
        # bias is computed of the relative position
        relative_pos_bias = self.mlp_bias(self.relative_pos).view(-1, self.n_heads) # ((2 * Wh - 1) * (2 * Ww - 1), n_heads)
        relative_pos_bias = rearrange(relative_pos_bias[self.relative_idx.flatten().to(torch.int32)], '(H1 H2 W1 W2) H -> 1 H (H1 W1) (H2 W2)', H1=self.wdw_h, H2=self.wdw_h,  
                                      W1=self.wdw_w, W2=self.wdw_w)
        relative_pos_bias = 16 * F.sigmoid(relative_pos_bias)
        
        # add the bias to the p matrix
        P += relative_pos_bias
        
        # calculate the attention matrix
        if mask is not None:
            S = (rearrange(P, '(B Nw) H N1 N2 -> B Nw H N1 N2', B=P.shape[0] // mask.shape[0], Nw=mask.shape[0]) + \
                rearrange(mask, 'Nw N1 N2 1 -> 1 Nw 1 N1 N2'))
            S = rearrange(S, 'B Nw H N1 N2 -> (B Nw) H N1 N2', B=P.shape[0] // mask.shape[0], Nw=mask.shape[0])
        S = F.softmax(P, dim=-1)
        
        # dropout layer
        P = self.attention_drop(P)
        
        # linear projection to value space
        A = torch.einsum('bhnm,bhmd->bhnd', P, V)
        A = rearrange(A, 'B H N D -> B N (H D)', H=self.n_heads, D=self.val_dim)
        
        return self.projection_drop(self.WA(A))
        
    def _calculate_relat_pos(self):
        zero_center_coord_h = torch.arange(-(self.wdw_h - 1), self.wdw_h, dtype=torch.float32)
        zero_center_coord_w = torch.arange(-(self.wdw_w - 1), self.wdw_w, dtype=torch.float32)
        
        coords = torch.cartesian_prod(zero_center_coord_h, zero_center_coord_w).view(
            2 * self.wdw_h - 1, 
            2 * self.wdw_w - 1, 
            2
        )
        return coords
    
    def _normalise_relat_pos(self, relative_pos, scale):
        relative_pos[..., 0] /= (self.wdw_h - 1)
        relative_pos[..., 1] /= (self.wdw_w - 1)
        return relative_pos * scale
    
    def _final_pos(self, normalised_pos, scale):
        log_base_scale = lambda x, scale: torch.sign(x) * torch.log2(torch.abs(x) + 1.0) / torch.log2(torch.tensor(scale))
        return log_base_scale(normalised_pos, scale)
    
    def _pos(self, scale):
        relative_pos = self._calculate_relat_pos()
        normalised_pos = self._normalise_relat_pos(relative_pos, scale)
        return self._final_pos(normalised_pos, scale)
    
    def _index(self):
        h_coords, w_coords = torch.arange(self.wdw_h, dtype=torch.float32), torch.arange(self.wdw_w, dtype=torch.float32)
        coords = torch.cartesian_prod(h_coords, w_coords).view(
            self.wdw_h, 
            self.wdw_w,
            2
        )
        coords = rearrange(coords, 'H W D -> D (H W)', H=self.wdw_h, W=self.wdw_w) # (2, wH*wW)
        relative_coords = coords.unsqueeze(-1) - coords.unsqueeze(-2) # (2, wH*wW, wH*wW)
        relative_coords = rearrange(relative_coords, 'D (H1 W1) (H2 W2) -> (H1 W1) (H2 W2) D', H1=self.wdw_h, W1=self.wdw_w, H2=self.wdw_h, W2=self.wdw_w) # (wH*wW, wH*wW, 2)
        for i, wdw_i in enumerate([self.wdw_h, self.wdw_w]):
            relative_coords[:, :, i] += wdw_i - 1
        relative_coords[:, :, 0] *= 2 * self.wdw_w - 1 # multiply by width range
        relative_coords = relative_coords.sum(-1)
        return relative_coords
    
class SwinTransformerBlock(nn.Module):
    def __init__(self, in_dim, n_heads, img_shape, wdw_size:int =4, shift:int =0, mlp_factor=4.0, bias=True, proj_drop=.0, att_drop=.0, drop_path=.0):
        super().__init__()
        self.in_dim = in_dim 
        self.n_heads = n_heads 
        self.wdw_size = wdw_size
        self.shift = shift 
        self.mlp_factor = mlp_factor
        self.bias = bias 
        self.proj_drop = proj_drop
        self.att_drop = att_drop 
        self.drop_path = drop_path
        self.img_shape = img_shape

        self.norm1 = nn.LayerNorm(self.in_dim)
        self.attn = WindowMSA(self.in_dim, (self.wdw_size, self.wdw_size), \
            n_heads=self.n_heads, att_drop=self.att_drop, proj_drop=self.proj_drop)
        
        self.drop_path = DropPath(drop_prob=self.drop_path) if self.drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.in_dim)
        
        self.mlp = MLP(self.in_dim, int(self.in_dim * mlp_factor), self.in_dim, self.proj_drop)
        
        mask = self._compute_mask(self.img_shape) if self.shift > 0 else None 
        self.register_buffer('attn_mask', mask)
            
    def forward(self, x):
        # x is (B, H, W, C)
        assert x.shape[1] == self.img_shape[0] and x.shape[2] == self.img_shape[1], f'({x.shape[1]},{self.img_shape[0]}), ({x.shape[2]},{self.img_shape[1]})'
        
        skip = x 
        # shift 
        s_x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2)) # roll for shift to get all possible patches
        
        # partition
        x_wdws = self._magic_partition(s_x, self.wdw_size, self.in_dim)
        
        # perform WMSA // SWMSA
        att_wdws = self.attn(x_wdws, mask=self.attn_mask) # B * n_wdws, wdw_size², C
        
        # merging
        att_wdws = self._magic_reverse(att_wdws, x.shape, self.wdw_size)
        
        # reverse shift
        s_x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        
        # perform skip and normalise
        x = skip + self.drop_path(self.norm1(x))
        
        # apply mlp
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def _compute_mask(self, img_shape):
        img_mask = self._calc_mask(self.wdw_size, self.shift, img_shape)
        wdw_partitioned_img = self._magic_partition(img_mask, self.wdw_size, 1).squeeze(-1)
        mask = rearrange(wdw_partitioned_img, 'NW W1W2C -> NW 1 W1W2C') - rearrange(wdw_partitioned_img, 'NW W1W2C -> NW W1W2C 1')
        mask = mask.masked_fill(mask != 0, -torch.inf).masked_fill(mask == 0, 0) # (num_wdw, wdw_size * wdw_size, wdw_size * wdw_size)
        return mask.unsqueeze(-1)
    
    def _magic_reverse(self, input, img_shape, wdw_size):
        return rearrange(input, '(B WH WW) (W1 W2) C -> B (WH W1) (WW W2) C', B=img_shape[0], WH=img_shape[1] // wdw_size, 
                         WW=img_shape[2] // wdw_size, W1=wdw_size, W2=wdw_size) # (B, H, W, C)
      
    def _magic_partition(self, input, wdw_size, in_dim):
        wdw_partitioned_img = self._wdw_partition(input, wdw_size)
        return rearrange(wdw_partitioned_img, 'B W1 W2 C -> B (W1 W2) C', W1=self.wdw_size, W2=self.wdw_size, C=in_dim)

    def _wdw_partition(self, input, wdw_size):
        b, h, w, c = input.shape
        assert h % wdw_size == 0 and w % wdw_size == 0, 'partition failed, height or width non div by wdw_size'
        wh, ww = h // wdw_size, w // wdw_size
        return rearrange(input, 'B (WH W1) (WW W2) C -> (B WH WW) W1 W2 C', B=b, WH=wh, W1=wdw_size, WW=ww, W2=wdw_size) # (num_wdw, wdw_size, wdw_size, c)
        
    def _calc_mask(self, wdw_size, shift, img_shape):
        h_indices = [[0, -wdw_size], [-wdw_size, -shift], [-shift, None]]
        w_indices = [[0, -wdw_size], [-wdw_size, -shift], [-shift, None]]

        hw_slices = product(h_indices, w_indices)

        img_mask = torch.zeros((1, *img_shape, 1), dtype=torch.float32)  # Initialize mask
        for cnt, (h, w) in enumerate(hw_slices):
            img_mask[:, slice(*h), slice(*w), :] = cnt
            
        return img_mask # (1, H, W, 1)

class PatchMerging(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim 
        
        self.norm = nn.LayerNorm(4 * self.in_dim)
        self.linear = nn.Linear(4 * self.in_dim, 2 * self.in_dim)
        
    def forward(self, x):
        B, H, W, C = x.shape 
        assert H % 2 == 0 and W % 2 == 0, 'either height or width not div by 2'
        assert x.shape[-1] == self.in_dim, f'{x.shape[-1]},{self.in_dim}'
        
        x = torch.cat([x[:, h_start::2, h_stop::2, :] for (h_start, h_stop) in product([0, 1], [0, 1])], dim=-1) # (B, H//2, W//2, C * 4)

        x = self.linear(self.norm(x))
        return x 

class PatchExpanding(nn.Module):
    def __init__(self, in_dim, dim_scale=2):
        super().__init__()
        self.in_dim = in_dim 
        self.dim_scale = dim_scale 
        
        assert self.in_dim % self.dim_scale == 0, 'in dim not div by dim scale'
        self.linear = nn.Linear(self.in_dim, self.in_dim * self.dim_scale, bias=False)
        self.norm = nn.LayerNorm(self.in_dim // self.dim_scale)
        
    def forward(self, x):
        B, H, W, C = x.shape # (B, H, W, C)
        assert self.in_dim == C, 'smth wrong with init of self.in_dim'
        
        x = self.linear(x) # (B, H, W, C * 2)
        x = rearrange(x, 'B H W (S1 S2 C) -> B (H S1) (W S2) C', B=B, H=H, W=W, S1=2, S2=2, C=(C // self.dim_scale)) # (B, H * 2, W * 2, C // 2)
        x = self.norm(x)
        
        return x 

class SwinTransformer(nn.Module):
    def __init__(self, in_dim, img_shape, depth, n_heads, wdw_size, mlp_factor:4., bias=True, proj_drop=0.0, att_drop=0.0, drop_path=0.0, downsample:int=0):
        super().__init__()
        self.in_dim = in_dim
        self.downsample = downsample
        self.img_shape = img_shape 
        self.depth = depth 
        self.n_heads = n_heads 
        self.wdw_size = wdw_size 
        self.mlp_factor = mlp_factor
        self.bias = bias 
        self.proj_drop = proj_drop 
        self.drop_path = drop_path 
        self.att_drop = att_drop
        
        self.swin_in_dim = 2 * self.in_dim if self.downsample == 0 else self.in_dim // 2 if self.downsample == 1 else self.in_dim
        self.swin_img_shape = (self.img_shape[0] // 2, self.img_shape[1] // 2) if self.downsample == 0 \
            else (self.img_shape[0] * 2, self.img_shape[1] * 2) if self.downsample == 1 else self.img_shape
                
        patch = PatchMerging(self.in_dim) if self.downsample == 0 else PatchExpanding(self.in_dim) if self.downsample == 1 else None
        
        arch_list = [] if patch is None else [patch]
        
        arch_list += [
            SwinTransformerBlock(in_dim=self.swin_in_dim, n_heads=self.n_heads, img_shape=self.swin_img_shape, wdw_size=self.wdw_size, 
                                 shift=(wdw_size // 2 if i % 2 else 0), mlp_factor=self.mlp_factor, bias=self.bias, proj_drop=self.proj_drop, 
                                 att_drop=self.att_drop, drop_path=self.drop_path[i]) for i in range(depth)
        ] # (B, H // 2, W // 2, C * 2)
        
        self.layer = nn.Sequential(*arch_list)
        
    def forward(self, x):
        return self.layer(x) 

class UDown(nn.Module):
    def __init__(self, embed_dim, nb_h_patches, nb_w_patches, depth_list_enc, n_heads_lower, wdw_size, mlp_factor, bias, proj_drop, att_drop, stochastic_depth):
        super().__init__()
        self.embed_dim = embed_dim 
        self.nb_h_patches = nb_h_patches
        self.nb_w_patches = nb_w_patches
        self.depth_list_enc = depth_list_enc
        self.n_heads_lower = n_heads_lower
        self.wdw_size = wdw_size
        self.mlp_factor = mlp_factor
        self.bias = bias 
        self.proj_drop = proj_drop
        self.att_drop = att_drop
        self.stochastic_depth = stochastic_depth
        
        assert len(self.depth_list_enc) == len(self.n_heads_lower)
        self.arch = nn.ModuleList([
            SwinTransformer(in_dim=self.embed_dim * 2 ** max(0, i - 1), 
                            img_shape=(self.nb_h_patches // 2 ** max(0, i - 1), self.nb_w_patches // 2 ** max(0, i - 1)), 
                            depth=depth, 
                            n_heads=n_heads, 
                            wdw_size=self.wdw_size, 
                            mlp_factor=self.mlp_factor, 
                            bias=self.bias, 
                            proj_drop=self.proj_drop, 
                            att_drop=self.att_drop, 
                            drop_path=self.stochastic_depth[sum(self.depth_list_enc[:i]):sum(self.depth_list_enc[:i+1])], # stochastic depth, as used in the impl details of paper
                            downsample=2 if i == 0 else 0
                            ) for i, (depth, n_heads) in enumerate(zip(self.depth_list_enc, self.n_heads_lower))
            ]) # (B, H, W, C) -> (B, H // 2, W // 2, C * 2) --> (B, H // 2**(len(list)-1), W // 2**(len(list)-1), C * 2**(len(list)-1))
        # (B, H // 2**(len(list)-1), W // 2**(len(list)-1), C * 2**(len(list)-1)) cat(-1) -> (B, H // 2**(len(list)-1), W // 2**(len(list)-1), C * 2**(len(list)))
        
        self.norm = nn.LayerNorm(self.embed_dim * 2 ** (len(self.depth_list_enc) - 1))
        
    def forward(self, x):
        skip_list = []
        for layer in self.arch:
            x = layer(x)
            skip_list.append(x)
        return self.norm(x), skip_list
    
class UUp(nn.Module):
    def __init__(self, embed_dim, nb_h_patches, nb_w_patches, depth_list_dec, n_heads_upper, wdw_size, mlp_factor, bias, proj_drop, att_drop, stochastic_depth):
        super().__init__()
        self.embed_dim = embed_dim
        self.nb_h_patches = nb_h_patches
        self.nb_w_patches = nb_w_patches
        self.depth_list_dec = depth_list_dec
        self.n_heads_upper = n_heads_upper
        self.wdw_size = wdw_size
        self.mlp_factor = mlp_factor
        self.bias = bias 
        self.proj_drop = proj_drop
        self.att_drop = att_drop
        self.stochastic_depth = stochastic_depth
        
        self.linear_arch = nn.ModuleList([
            nn.Linear(self.embed_dim * (2 ** i), self.embed_dim * (2 ** (i - 1))) for i in range(len(self.depth_list_dec), 0, -1)
        ])
        
        self.up_arch = nn.ModuleList([
            SwinTransformer(
                in_dim=self.embed_dim * 2 ** i, 
                img_shape=(self.nb_h_patches // 2 ** i, self.nb_w_patches // 2 ** i), 
                depth = self.depth_list_dec[i - 1], 
                n_heads=self.n_heads_upper[i - 1], 
                wdw_size=self.wdw_size, 
                mlp_factor=self.mlp_factor, 
                bias=self.bias, 
                proj_drop=self.proj_drop, 
                att_drop=self.att_drop, 
                drop_path=self.stochastic_depth[sum(self.depth_list_dec[:(i-1)]):sum(self.depth_list_dec[:i])], 
                downsample=1
            ) for i in range(len(self.depth_list_dec), 0, -1)
        ])
        
        self.final_norm = nn.LayerNorm(self.embed_dim)
        
    def forward(self, x, skip_list):
        for layer, skip, linear in zip(self.up_arch, skip_list[::-1], self.linear_arch):
            x = layer(x)
            x = torch.cat([x, skip], dim=-1)
            x = linear(x)
        return self.final_norm(x) 

class SwinUNET(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, img_size=224, patch_size=4, wdw_size=7, embed_dim=96, depth_list_enc=[2, 2, 2], depth_list_dec=[2, 2, 2], depth_bneck=2, 
                 n_heads=[2, 3, 6, 12], mlp_factor=2.0, bias=True, proj_drop=0.0, att_drop=0.0, drop_path=0.2):
        super().__init__()
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.img_size = img_size 
        self.patch_size = patch_size
        self.wdw_size = wdw_size
        self.embed_dim = embed_dim 
        self.depth_list_enc = depth_list_enc
        self.depth_list_dec = depth_list_dec 
        self.depth_bneck = depth_bneck 
        self.n_heads = n_heads 
        self.mlp_factor = mlp_factor
        self.bias = bias 
        self.proj_drop = proj_drop
        self.att_drop = att_drop 
        self.drop_path = drop_path
        self.n_heads_lower = n_heads[:-1]
        self.n_heads_upper = self.n_heads_lower[::-1]
        self.n_heads_bneck = n_heads[-1] 
        
        
        self.patch_embed = PatchEmbed(
            in_dim=self.in_dim, img_h=self.img_size, img_w=self.img_size, patch_h=self.patch_size, patch_w=self.patch_size, embed_dim=self.embed_dim
        )
        self.n_patches = self.patch_embed.n_patches
        self.nb_h_patches, self.nb_w_patches = self.patch_embed.nb_h_patches, self.patch_embed.nb_w_patches
        self.pos_embed = nn.Parameter(torch.arange(self.n_patches * self.embed_dim, dtype=torch.float32).reshape(1, self.n_patches, self.embed_dim))
        self.pos_dropout = nn.Dropout(self.proj_drop)
    
        self.stochastic_depth = np.linspace(start=0, stop=self.drop_path, num=np.sum(self.depth_list_enc + [self.depth_bneck])).tolist()        
            
        self.down_arch = UDown(embed_dim=self.embed_dim, nb_h_patches=self.nb_h_patches, nb_w_patches=self.nb_w_patches, depth_list_enc=self.depth_list_enc, 
                               n_heads_lower=self.n_heads_lower, wdw_size=self.wdw_size, mlp_factor=self.mlp_factor, bias=self.bias, proj_drop=self.proj_drop, att_drop=self.att_drop, 
                               stochastic_depth=self.stochastic_depth)
        
        self.bneck = SwinTransformer(in_dim=self.embed_dim * 2 ** (len(self.depth_list_dec) - 1), img_shape=(self.nb_h_patches // 2 ** (len(self.depth_list_dec) - 1), self.nb_w_patches // 2 ** (len(self.depth_list_dec) - 1)), 
                            depth=self.depth_bneck, n_heads=self.n_heads_bneck, wdw_size=self.wdw_size, mlp_factor=self.mlp_factor, bias=self.bias, 
                            proj_drop=self.proj_drop, att_drop=self.att_drop, drop_path=self.stochastic_depth[sum(self.depth_list_enc):sum((self.depth_list_enc + [self.depth_bneck]))])
        
        self.up_arch = UUp(embed_dim=self.embed_dim, nb_h_patches=self.nb_h_patches, nb_w_patches=self.nb_w_patches, depth_list_dec=self.depth_list_dec, 
                           n_heads_upper=self.n_heads_upper, wdw_size=self.wdw_size, mlp_factor=self.mlp_factor, bias=self.bias, proj_drop=self.proj_drop, 
                           att_drop=self.att_drop, stochastic_depth=self.stochastic_depth)
        
        self.final_linear = nn.Linear(self.embed_dim, self.patch_size * self.patch_size)
        
    def forward(self, x):
        x = self.pos_dropout(self.patch_embed(x) + self.pos_embed) # (B, N, C)
        x = rearrange(x, 'B (N1 N2) C -> B N1 N2 C', N1=self.nb_h_patches, N2=self.nb_w_patches)
        x, skip_list = self.down_arch(x)
        x = self.bneck(x)
        x = self.up_arch(x, skip_list)
        x = self.final_linear(x)
        x = rearrange(x, 'B N1 N2 (H1 W1 C) -> B C (N1 H1) (N2 W1)', C=self.in_dim, H1=self.patch_size, W1=self.patch_size)
        # this part is for segmentation
        x = torch.sigmoid(x)
        return x

def test_swin_unet():
    swinunet = SwinUNET()
    x = torch.randn(10, 1, 224, 224)
    
    print(swinunet(x).shape)
    
def test_patch_expand():
    p = PatchExpanding(8)
    x = torch.randn(20, 8, 8, 8)
    print(p(x).shape)

def test_patch_merging():
    p = PatchMerging(4)
    x = torch.randn(20, 256, 256, 4)
    print(p(x).shape)

def test_swin_block():
    swin = SwinTransformerBlock(in_dim=3, n_heads=5, img_shape=(256, 256), wdw_size=4, shift=10)
    x = torch.randn((10, 256, 256, 3))
    print(swin(x).shape)

def test_mlp():
    mlp = MLP(100, 2000, 1000)
    x = torch.randn(50, 100)
    out = mlp(x)
    print(out.shape)

def test_window_msa():
    window_size = (4, 4)
    wdw = WindowMSA(1, window_size, n_heads=3)
    E = torch.randn(500, window_size[0] * window_size[1], 1)
    print(wdw(E).shape)
    
def test_swin_transformer():
    swin = SwinTransformer(in_dim=20, img_shape=(20, 20), depth=2, n_heads=2, wdw_size=4, mlp_factor=4.0, downsample=1)
    x = torch.randn(20, 20, 20, 20)
    print(swin(x).shape)
    
    

    


    