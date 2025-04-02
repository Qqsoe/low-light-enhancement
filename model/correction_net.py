import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os
from model.detail import Mlp

class CAFM(nn.Module):
    def __init__(self):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.conv_f1 = nn.Conv2d(16, 64, 1)  # Added to transform f1 to have 64 channels
        self.conv_f2 = nn.Conv2d(64, 64, 1)  # Added to ensure f2 has 64 channels

        self.avg1 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(64, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, 64, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c1, h1, w1 = f1.size()
        _, c2, h2, w2 = f2.size()

        f1 = self.conv_f1(f1)  # Transform f1 to have 64 channels
        f2 = self.conv_f2(f2)  # Ensure f2 has 64 channels

        f1 = f1.view(b, 64, -1)
        f2 = f2.view(b, 64, -1)

        avg_1 = torch.mean(f1, dim=-1, keepdim=True)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)

        avg_1 = F.relu(self.avg1(avg_1.unsqueeze(-1)))
        max_1 = F.relu(self.max1(max_1.unsqueeze(-1)))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)

        avg_2 = F.relu(self.avg2(avg_2.unsqueeze(-1)))
        max_2 = F.relu(self.max2(max_2.unsqueeze(-1)))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.view(b, 64, h1, w1)
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.view(b, 1, h1 * w1)
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.view(b, 64, h2, w2)
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.view(b, 1, h2 * w2)
        a2 = F.softmax(a2, dim=-1)

        f1 = f1.view(b, 64, h1, w1)
        f2 = f2.view(b, 64, h2, w2)

        if h1 != h2 or w1 != w2:
            f1 = F.interpolate(f1, size=(h2, w2), mode='bilinear', align_corners=False)

        f = f1 + f2  # Element-wise addition along the spatial dimensions
        f = f.view(b, 64, h2, w2)  # reshaping to the desired output shape

        return f

class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape

        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x+ self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def visualize_feature_maps(self, feature_maps):
        """
        可视化卷积层输出的特征图
        """
        # 确保从 GPU 转移到 CPU，并转换为 NumPy 数组
        feature_map_data_list = [feature_maps[0, i].detach().cpu().numpy() for i in range(feature_maps.shape[1])]
        feature_map_data_list = feature_map_data_list[:3]
        num_features = len(feature_map_data_list)
        cols = 4  # 每行显示4个特征图
        rows = (num_features + cols - 1) // cols  # 计算总行数

        # 调整图像大小，增大图片尺寸
        plt.figure(figsize=(8, 4 * rows))  # 增大整体图片
        for i, feature_map_data in enumerate(feature_map_data_list):
            plt.subplot(1, len(feature_map_data_list), i + 1)
            plt.imshow(feature_map_data, cmap="Blues")
            plt.title(f"{i + 1}")

            plt.axis('off')
        plt.tight_layout()
        plt.show()
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(

            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),

            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.cafm = CAFM()


    def forward(self, x, img2):
       x3= self.proj(x)
       x=self.cafm(img2,x3)

       return x


class Correct_net(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_heads=4, type='exp'):
        super(Correct_net, self).__init__()
        if type == 'exp':
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=False) # False in exposure correction
        else:
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)  # basic color matrix


        self.conv_large = conv_embedding(in_channels, out_channels)
        self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
        self.gamma_linear = nn.Linear(out_channels, 1)
        self.color_linear = nn.Linear(out_channels, 1)
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x,img2):

        x = self.conv_large(x,img2)

        x = self.generator(x)
        gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
        gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
        color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base

        return gamma, color


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    img = torch.Tensor(8, 3, 400, 600)
    global_net = Correct_net()
    gamma, color = global_net(img)
    print(gamma.shape, color.shape)