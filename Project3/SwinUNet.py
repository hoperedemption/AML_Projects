import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from math import sqrt, log
from einops import rearrange
from itertools import product

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
        # (B*n_patches, N, C) -> (B*n_patches, N, n_heads * d)
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
    def __init__(self, in_dim, n_heads, img_shape, wdw_size=4, shift=0, mlp_factor=4.0, bias=True, drop=.0, att_drop=.0, drop_path=.0):
        super().__init__()
        self.in_dim = in_dim 
        self.n_heads = n_heads 
        self.img_shape = img_shape
        self.wdw_size = wdw_size
        self.shift = shift 
        self.mlp_factor = mlp_factor
        self.bias = bias 
        self.drop = drop
        self.att_drop = att_drop 
        self.drop_path = drop_path
        
    
    def _wdw_partition(self, input, wdw_size):
        assert h % wdw_size == 0 and w % wdw_size == 0, 'partition failed, height or width non div by wdw_size'
        b, h, w, c = input.shape
        wh, ww = h // wdw_size, w // wdw_size
        return rearrange(input, 'B (WH W1) (WW W2) C -> (B WH WW) W1 W2 C', B=b, WH=wh, W1=wdw_size, WW=ww, W2=wdw_size)  
        
    def _calc_mask(self, wdw_size, shift, img_shape):
        h_indices = [[0, -wdw_size], [-wdw_size, -shift], [-shift, None]]
        w_indices = [[0, -wdw_size], [-wdw_size, -shift], [-shift, None]]

        hw_slices = product(h_indices, w_indices)

        img_mask = torch.zeros((1, *img_shape, 1), dtype=torch.float32)  # Initialize mask
        for cnt, (h, w) in enumerate(hw_slices):
            img_mask[:, slice(*h), slice(*w), :] = cnt
            
        return img_mask


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
    
test_window_msa()
    

    


    