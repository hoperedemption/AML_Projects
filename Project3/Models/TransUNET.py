import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from math import sqrt
from einops import rearrange
from torchvision.ops import stochastic_depth

def patchify(image, num_patches):  # (B, C, H, W)
    image_height, image_width = image.shape[-2], image.shape[-1]
    p1, p2 = int(sqrt(num_patches)), int(sqrt(num_patches))
    patch_height, patch_width = image_height // p1, image_width // p2
    
    image = rearrange(image, 'b c (p1 h) (p2 w) -> b p1 p2 c h w', h=patch_height, w=patch_width, p1=p1, \
        p2=p2, c=image.shape[1])
    return image       

class ConvBlock(nn.Sequential):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, act=nn.ReLU, dropout=0.1):
        super().__init__(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
            nn.BatchNorm2d(num_features=out_dim), 
            act(), 
            nn.Dropout(dropout)
        )

class DoubleConv(nn.Sequential):
    def __init__(self, in_dim, inter_dim, out_dim, kernel_size, stride, padding, act=nn.ReLU, dropout=0.1):
        super().__init__(
            ConvBlock(in_dim=in_dim, out_dim=inter_dim, kernel_size=kernel_size, stride=stride, padding=padding, act=act, dropout=dropout), 
            ConvBlock(in_dim=inter_dim, out_dim=out_dim, kernel_size=kernel_size, stride=stride, padding=padding, act=act, dropout=dropout)
        )
        
class DoubleBaseConv(nn.Sequential):
    def __init__(self, in_dim, out_dim, act=nn.ReLU, dropout=0.1):
        super().__init__(
            DoubleConv(in_dim, out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), act=act, dropout=dropout), 
            ConvBlock(out_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), act=act, dropout=dropout)
        )
        
class SkipBlock(nn.Module):
    def __init__(self, in_dim, out_dim, layer: nn.Module, drop_path=0.1):
        super().__init__()
        
        self.shortcut = ConvBlock(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if in_dim != out_dim else None
        self.layer = layer
        self.drop_path = drop_path
        self.stochastic_depth = stochastic_depth
        
    def forward(self, x):
        residual = x 
        x = self.layer(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        return residual + self.stochastic_depth(x, p=self.drop_path, mode='row')
        
    
class FusedMBConv(nn.Sequential):
    def __init__(self, in_dim, out_dim, expansion, dropout=0.1, act=nn.ReLU):
        exp_dim = in_dim * expansion
        super().__init__(
            nn.Sequential(
                SkipBlock(
                    in_dim=in_dim, 
                    out_dim=out_dim, 
                    layer=nn.Sequential(
                        ConvBlock(in_dim, exp_dim, act=nn.ReLU6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout=dropout), 
                        ConvBlock(exp_dim, out_dim, act=nn.Identity, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dropout=dropout)
                    ), drop_path=dropout
                ), 
                act()
            )
        )

class DoubleBaseConvFused(nn.Sequential):
    def __init__(self, in_dim, out_dim, expansion, dropout=0.1, act=nn.ReLU):
        super().__init__(
            FusedMBConv(in_dim, out_dim, expansion, dropout=dropout, act=act), 
            ConvBlock(out_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout=dropout, act=act)
        )     
        

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, input_dim, key_dim, val_dim, dropout=0.1, num_heads=1):
        super().__init__()
        
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.key_dim = key_dim 
        self.query_dim = key_dim 
        self.val_dim = val_dim 
        
        self.WQ = nn.Linear(input_dim, num_heads * self.query_dim, bias=False)
        self.WK = nn.Linear(input_dim, num_heads * self.key_dim, bias=False)
        self.WV = nn.Linear(input_dim, num_heads * self.val_dim, bias=False)
        self.WA = nn.Linear(num_heads * self.val_dim, input_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, E): # (B, N, E)
        # perform embedding projections
        Q, K, V = self.WQ(E), self.WK(E), self.WV(E) # (B, N, H * D)

        # rearange the matrices
        Q = rearrange(Q, 'B N (H D) -> B H N D', H=self.num_heads, D=self.query_dim)
        K = rearrange(K, 'B N (H D) -> B H N D', H=self.num_heads, D=self.key_dim)
        V = rearrange(V, 'B N (H D) -> B H N D', H=self.num_heads, D=self.val_dim)
        
        # compute attention matrix
        P = torch.einsum('bhmd,bhnd->bhmn', Q, K)
        
        # compute attention values
        S = torch.softmax(P / sqrt(self.key_dim), dim=-1)
        S = self.attn_dropout(S)
        
        # project with attention weights
        A = torch.einsum('bhnm,bhmd->bhnd', S, V)
        A = rearrange(A, 'B H N D -> B N (H D)', H=self.num_heads, D=self.val_dim)
        A = self.proj_dropout(A)
        
        return self.WA(A) 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))
    
class TransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_key_dim, hidden_val_dim, hidden_mlp_dim, num_heads=1, dropout=0.1, drop_path=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_key_dim = hidden_key_dim
        self.hidden_val_dim = hidden_val_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.self_attention = MultiHeadedSelfAttention(
            input_dim=self.input_dim, key_dim=self.hidden_key_dim, 
            val_dim=self.hidden_val_dim, dropout=self.dropout, num_heads=self.num_heads
        )
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.mlp = MLP(input_dim, hidden_mlp_dim)
        self.norm2 = nn.LayerNorm(self.input_dim)
        
        self.drop_path = drop_path
        self.stochastic_depth = stochastic_depth
    
    def forward(self, x):
        # First Residual Connection
        residual = x 
        x = self.norm1(x)
        att_out = self.self_attention(x)
        x = residual + self.stochastic_depth(att_out, self.drop_path, mode='row')
        
        # Second Residual Connection
        residual = x 
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.stochastic_depth(x, self.drop_path, mode='row')
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, T, device):
        assert d_model % 2 == 0, f'{d_model} not div by 2'
        with torch.no_grad():
            self.pe = torch.zeros((T, d_model), device=device)
            position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model // 2, dtype=torch.float32) * (-torch.log(torch.tensor(1e4)) / d_model)).to(device).unsqueeze(0)
            
            self.pe[:, 0::2] = torch.sin(position * div_term).to(device)
            self.pe[:, 1::2] = torch.cos(position * div_term).to(device)
            
class Transformer(nn.Module):
    def __init__(self, device, num_tokens, input_dim, hidden_key_dim, hidden_val_dim, hidden_mlp_dim, output_dim, max_length, num_layers=1, num_heads=1):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.hidden_key_dim = hidden_key_dim
        self.hidden_val_dim = hidden_val_dim
        self.hidden_mlp_dim = hidden_mlp_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.max_length = max_length
        self.num_heads = num_heads
        self.device = device
                
        # register buffer
        pe = PositionalEncoding(self.input_dim, max_length, self.device).pe
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.arch = nn.Sequential(
            *[TransformerLayer(input_dim=self.input_dim, hidden_key_dim=self.hidden_key_dim,
                               hidden_val_dim=self.hidden_val_dim, hidden_mlp_dim=self.hidden_mlp_dim, 
                               num_heads=self.num_heads, drop_path=0.1*i, dropout=0.1*i) for i in range(num_layers)]
        )
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        x = self.arch(x)
        return self.norm(self.linear(x))
    
class Encoder(nn.Module):
    def __init__(self, in_dim, image_shape, n_patches, device, feature_list=[32, 64, 128]):
        super().__init__()
        
        feature_list = [in_dim] + feature_list
        self.depth = len(feature_list) - 1
        self.feature_learning = nn.ModuleList(
            [
                # DoubleBaseConv(in_channel, out_channel) 
                DoubleBaseConvFused(in_channel, out_channel, 4, dropout=0.1 * i, act=nn.SiLU) \
                for i, (in_channel, out_channel) in enumerate(zip(feature_list[:-1], feature_list[1:]))
            ]
        )
        

        self.n_patches = n_patches
        sqr = int(sqrt(self.n_patches))
        self.n_patches_h, self.n_patches_w = sqr, sqr 
        self.patch_hid_h, self.patch_hid_w = image_shape[-2] // 2 ** self.depth, image_shape[-1] // 2 ** self.depth
        self.patches_h, self.patches_w = self.patch_hid_h // self.n_patches_h, self.patch_hid_w // self.n_patches_w
        self.depth_channels = feature_list[-1]

        self.transformer_in_dim = self.depth_channels * self.patches_h * self.patches_w
        self.transformer = Transformer(device=device, num_tokens=self.n_patches, input_dim=self.transformer_in_dim, 
                                       hidden_key_dim=512, hidden_val_dim=512, hidden_mlp_dim=1024, max_length=self.n_patches, 
                                       output_dim=self.transformer_in_dim, num_layers=5, num_heads=8)

        self.bneck = ConvBlock(in_dim=self.depth_channels * self.n_patches, out_dim=feature_list[-1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    def forward(self, x):
        skip = x
        skip_list = []
        
        for block in self.feature_learning:
            skip = block(skip)
            skip_list.append(skip)
        
        b, c, h, w = skip.shape
        p = patchify(skip, self.n_patches)
        p = rearrange(p, 'b p1 p2 c h1 h2 -> b (p1 p2) (c h1 h2)')
        p = self.transformer(p)
        p = rearrange(p, 'b (p1 p2) (c h1 h2) -> b (c p1 p2) h1 h2', p1=self.n_patches_h, p2=self.n_patches_w, 
                      c=c, h1=self.patches_h, h2=self.patches_w)
      
        return self.bneck(p), skip_list
        
class Decoder(nn.Module):
    def __init__(self, out_dim, feature_list=[128, 64, 32]):
        super().__init__()
        
        self.up_arch = nn.ModuleList([
            nn.ModuleList(
                [nn.Upsample(scale_factor=2, mode='bilinear'), 
                 DoubleConv(in_channels * 2, in_channels, in_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ]
            ) for in_channels in feature_list
        ])
        
        self.final_seg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(feature_list[-1] // 2, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.Sigmoid()
        )
        
    def forward(self, x, skip_list):
        up = x 
        for skip, block in zip(skip_list[::-1], self.up_arch):
            up = block[0](up)
            up = torch.cat([skip, up], dim=1)
            up = block[1](up)
        
        return self.final_seg(up)
    
class TransUNet(nn.Module):
    def __init__(self, in_dim, out_dim, image_shape, n_patches, device, feature_list_down=[32, 64, 128], feature_list_up=[128, 64, 32]):
        super().__init__()
        
        self.down = Encoder(in_dim=in_dim, image_shape=image_shape, n_patches=n_patches, device=device, feature_list=feature_list_down)
        self.up = Decoder(out_dim=out_dim, feature_list=feature_list_up)
        
    def forward(self, x):
        x, skip = self.down(x)
        return self.up(x, skip)
        
        
def test_conv_block():
    t = torch.randn(50, 1, 112, 112)
    conv = ConvBlock(1, 20, (1, 1), (1, 1), (0, 0))
    print(conv(t).shape)
    
def test_conv_base():
    t = torch.randn(50, 1, 112, 112)
    conv = DoubleBaseConv(1, 10)
    print(conv(t).shape)
    
def test_transformer():
    c, h, w = 3, 32, 32
    t = torch.randn(50, c, h, w)
    n_patches = 16
    n_patches_h, n_patches_w = int(n_patches // sqrt(n_patches)), int(n_patches // sqrt(n_patches))
    patch_h, patch_w = int(h // n_patches_h), int(w // n_patches_w)
    p = patchify(t, n_patches)
    p = rearrange(p, 'b p1 p2 c h1 h2 -> b (p1 p2) (c h1 h2)')
    input_dim = c * patch_h * patch_w
    trans = Transformer('cpu', n_patches, input_dim, input_dim * 3, input_dim * 3, input_dim, n_patches, num_layers=3, num_heads=2)
    out = trans(p)
    
    out = rearrange(out, 'b (p1 p2) (c h1 h2) -> b c (p1 h1) (p2 h2)', p1=n_patches_h, p2=n_patches_w, h1=patch_h, h2=patch_w)
    print(out.shape)
    
def test_encoder():
    in_dim = 4
    n_patches = 4 
    t = torch.randn(50, in_dim, 112, 112)
    enc = Encoder(in_dim=in_dim, image_shape=(112, 112), n_patches=n_patches, device='cpu', feature_list=[32, 64, 128])
    print(enc(t)[0].shape)
    
def test_fused_block():
    t = torch.randn(50, 3, 64, 64)
    fused = FusedMBConv(in_dim=3, out_dim=6, expansion=4)
    print(fused)
    print(fused(t).shape)

def test_transunet():
    in_dim = 4
    n_patches = 4 
    t = torch.randn(50, in_dim, 112, 112)
    unet = TransUNet(in_dim, 1, (112, 112), n_patches, 'cpu')
    out = unet(t)
    print(out.shape)
    print(out.min(), out.max())
    
