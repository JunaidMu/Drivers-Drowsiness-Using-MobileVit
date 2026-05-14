import torch
import torch.nn as nn

def conv_1x1_bn(inp, oup):
    # quick 1x1 conv block
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    # standard conv block
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, kernal_size//2, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(MV2Block, self).__init__()
        self.stride = stride
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expansion != 1:
            # expand channels first
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
        
        layers.extend([
            # depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # project back down
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # skip connection if shapes match
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_sz, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # first norm
        self.ln1 = nn.LayerNorm(embed_dim)
        # self-attention bit
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout)
        
        # second norm
        self.ln2 = nn.LayerNorm(embed_dim)
        # tiny feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ffn_latent_sz),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_latent_sz, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # transformer expects (seq_len, batch, dim)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x
    

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, ffn_dim, patch_size=(2, 2), num_transformer_blocks=2):
        super(MobileViTBlock, self).__init__()
        self.patch_h, self.patch_w = patch_size

        # local CNN features
        self.local_rep = nn.Sequential(
            conv_nxn_bn(in_channels, in_channels),
            nn.Conv2d(in_channels, transformer_dim, 1, 1, 0, bias=False)
        )

        # transformer for global context
        self.global_rep = nn.Sequential(*[
            TransformerEncoder(transformer_dim, ffn_dim) for _ in range(num_transformer_blocks)
        ])

        # merge everything back
        self.proj = nn.Conv2d(transformer_dim, in_channels, 1, 1, 0, bias=False)
        self.fusion = conv_nxn_bn(2 * in_channels, in_channels)

    def forward(self, x):
        res = x.clone()

        # grab local features first
        x = self.local_rep(x)

        # break feature map into patches
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_h, self.patch_h).unfold(3, self.patch_w, self.patch_w)

        # reshape for transformer
        new_h, new_w = H // self.patch_h, W // self.patch_w
        x = x.permute(0, 4, 5, 2, 3, 1).reshape(B, self.patch_h * self.patch_w, new_h * new_w, C)
        
        # make patches the sequence
        x = x.transpose(1, 2)
        
        processed_batches = []
        for b in range(B):
            # process each batch separately
            processed_batches.append(self.global_rep(x[b]))
        
        x = torch.stack(processed_batches, dim=0)
        x = x.transpose(1, 2)

        # fold patches back into image form
        x = x.reshape(B, self.patch_h, self.patch_w, new_h, new_w, C)
        x = x.permute(0, 5, 3, 1, 4, 2).reshape(B, C, H, W)
        
        # fuse local + global info
        x = self.proj(x)
        x = self.fusion(torch.cat((res, x), dim=1))
        return x

class MobileViT_XXS(nn.Module):
    def __init__(self, num_classes=2): 
        super(MobileViT_XXS, self).__init__()
        
        # first conv layer
        self.conv1 = conv_nxn_bn(3, 16, kernal_size=3, stride=2)
        
        # early local feature extraction
        self.mv2_1 = nn.Sequential(
            MV2Block(16, 16, stride=1, expansion=2),
            MV2Block(16, 24, stride=2, expansion=2),
            MV2Block(24, 24, stride=1, expansion=2),
            MV2Block(24, 24, stride=1, expansion=2),
        )
        self.mv2_2 = MV2Block(24, 48, stride=2, expansion=2)
        
        # first mobilevit stage
        self.mvit_1 = MobileViTBlock(48, transformer_dim=64, ffn_dim=128, num_transformer_blocks=2)
        
        self.mv2_3 = MV2Block(48, 64, stride=2, expansion=2)
        self.mvit_2 = MobileViTBlock(64, transformer_dim=80, ffn_dim=160, num_transformer_blocks=4)
        
        self.mv2_4 = MV2Block(64, 80, stride=2, expansion=2)
        self.mvit_3 = MobileViTBlock(80, transformer_dim=96, ffn_dim=192, num_transformer_blocks=3)
        
        # final classifier
        self.conv2 = conv_1x1_bn(80, 320)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.mvit_1(x)
        x = self.mv2_3(x)
        x = self.mvit_2(x)
        x = self.mv2_4(x)
        x = self.mvit_3(x)
        x = self.conv2(x)
        x = self.pool(x).view(-1, 320)
        return self.fc(x)
    

print(MobileViT_XXS()(torch.randn(1, 3, 256, 256)).shape)