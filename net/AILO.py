from einops import einops
from net.transformer_utils import *

class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.DDconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim // 4),
            nn.Conv2d(dim, dim, 5, stride=1, padding=(5 // 2) * 2, groups=dim // 4, dilation=2),
        )
        self.DDconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim // 4),
            nn.Conv2d(dim, dim, 7, stride=1, padding=(7 // 2) * 3, groups=dim // 4, dilation=3),
        )
        self.DDconv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 7, 1, 3, groups=dim // 4),
            nn.Conv2d(dim, dim, 9, stride=1, padding=(9 // 2) * 4, groups=dim // 4, dilation=4),
        )

        self.act = nn.Tanh()

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * dim, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        res_x = x.clone()
        x = self.norm(x)

        x0 = self.act(x)
        x1 = self.DDconv1(x) * x0
        x2 = self.DDconv2(x) * x0
        x3 = self.DDconv3(x) * x0

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion(x)

        return x + res_x

class FeatureSampler(nn.Module):
    def __init__(
            self,
            in_channels,  
            num_points, 
            region_size,  
            offset_conv_ksize=3,  
            offset_range_factor=1.0, 
            n_groups=1, 
            stride=1  
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_points = num_points
        self.region_size = region_size
        self.offset_range_factor = offset_range_factor
        self.n_groups = n_groups
        self.stride = stride

        assert in_channels % n_groups == 0, "in_channels must be divisible by n_groups"
        self.n_group_channels = in_channels // n_groups

        offset_conv_input_channels = 1 + in_channels

        self.conv_offset = nn.Sequential(
            nn.Conv2d(offset_conv_input_channels, offset_conv_input_channels,
                      offset_conv_ksize, stride, offset_conv_ksize // 2,
                      groups=offset_conv_input_channels),
            nn.GroupNorm(1, offset_conv_input_channels),
            nn.GELU(),
            nn.Conv2d(offset_conv_input_channels, 2 * num_points, 1, 1, 0, bias=False)
        )

        self._init_weights()

    def _init_weights(self):
        if hasattr(self.conv_offset[-1], 'weight'):
            nn.init.constant_(self.conv_offset[-1].weight, 0)

    @torch.no_grad()
    def _get_ref_points(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0) 
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0) 
        ref = ref[None, ...].expand(B, -1, -1, -1) 

        return ref

    def forward(self, intensity, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        offset_input = torch.cat([intensity, x], dim=1)
        offset = self.conv_offset(offset_input)
        offset = offset.view(B, self.num_points, 2, H, W)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (H - 1.0), 1.0 / (W - 1.0)],
                                        device=device).reshape(1, 1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        reference = self._get_ref_points(H, W, B, dtype, device)
        reference = reference.unsqueeze(1)  # [B, 1, H, W, 2]
        pos = offset.permute(0, 1, 3, 4, 2) + reference  # [B, num_points, H, W, 2]

        sampled_features = []
        for i in range(self.num_points):
            sampled = F.grid_sample(
                x,
                pos[:, i, :, :, :],
                mode='bilinear',
                align_corners=True,
                padding_mode='border'
            )
            sampled_features.append(sampled)

        sampled_features = torch.cat(sampled_features, dim=1)

        if self.region_size != H or self.region_size != W:
            start_h = (H - self.region_size) // 2
            start_w = (W - self.region_size) // 2
            sampled_features = sampled_features[
                               :, :,
                               start_h:start_h + self.region_size,
                               start_w:start_w + self.region_size
                               ]

        return sampled_features

class AILO(nn.Module):
    def __init__(self, in_channels, out_channels, num_points=3, region_size=32, kernel_size=5, groups=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.region_size = region_size
        self.kernel_size = kernel_size
        self.num_points = num_points

        self.sampler = FeatureSampler(
            in_channels=in_channels,
            num_points=num_points,
            region_size=region_size,
            offset_range_factor=1.0,
            n_groups=1,
            stride=1
        )

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels // groups)
        self.key_conv = nn.Conv2d(in_channels * num_points, in_channels, 1, groups=in_channels // groups)
        self.value_conv = nn.Conv2d(in_channels * num_points, in_channels, 1, groups=in_channels // groups)

        self.norm = LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels // self.groups, self.kernel_size ** 2)

        self.FFN = FFN(out_channels)

    def _get_dynamic_kernel(self, x, intensity):
        B, C, H, W = x.shape

        patches = self.sampler(intensity, x)
        patches = patches.view(B, self.num_points * C, self.region_size, self.region_size)
        K = self.key_conv(patches)
        K = K.view(B, self.groups, C // self.groups, self.region_size * self.region_size)  # [B,G,Cg,SS]

        Q = self.query_conv(x)
        Q = Q.view(B, self.groups, C // self.groups, H * W)  # [B,G,Cg,HW]

        affinity = torch.einsum('bgcj,bgci->bgij', Q, K) / (C // self.groups) ** 0.5  # [B,G,SS,HW]
        affinity = F.softmax(affinity, dim=-1)  # [B,G,SS,HW]

        dynamic_weight = self.value_conv(patches)  # B C S S
        dynamic_weight = dynamic_weight.view(B, self.groups, self.region_size**2, C // self.groups)  # B G SS C/G
        dynamic_weight = self.fc(dynamic_weight)  # B G SS KK
        kernels = torch.einsum('bgij,bgik->bgjk', affinity, dynamic_weight)  # [B,G,HW,KK] = [B,G,SS,HW] * [B G,SS,KK]

        kernels = kernels.permute(0, 1, 3, 2).contiguous()  # [B,G,KK,HW]
        kernels = kernels.view(B, self.groups, H * W, self.kernel_size ** 2)

        return kernels

    def forward(self, x, intensity):
        B, C, H, W = x.shape
        res_x = x.clone()
        x = self.norm(x)

        kernels = self._get_dynamic_kernel(x, intensity)  # [B,G,HW,KK]

        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        x_unfold = x_unfold.view(B, self.groups, H * W, -1, self.kernel_size ** 2)  # [B,G,HW,Cg,KK]

        low_freq = torch.einsum('bgik,bgijk->bgij', kernels, x_unfold)
        low_freq = low_freq.reshape(B, -1, H, W)  # [B,C,H,W]

        low_freq = low_freq + res_x

        low_freq = self.FFN(low_freq)

        return low_freq
