import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.encoder_decoder import Decoder
from net.i_hv_encoder import i_hv_encoder
from net.quantizer import VectorQuantizer
from net.i_decoder import i_decoder

class Context_Aware_Module(nn.Module):
    def __init__(self, in_channels):
        super(Context_Aware_Module, self).__init__()
        self.in_channels = in_channels
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1) 
        self.conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, in_channels, 7, padding=3)
        self.fusionlayer = nn.Conv2d(in_channels * 3, in_channels, 3, padding=1)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
        self.act = nn.Tanh()

    def forward(self, x):
        x0 = self.act(x)
        f3 = self.conv3(x) * x0
        f5 = self.conv5(x) * x0
        f7 = self.conv7(x) * x0
        multi_features = torch.cat([f3, f5, f7], dim=1)
        fused_repair = self.fusionlayer(multi_features)
        
        gate_weights = self.gate_net(x) 
        
        out = gate_weights * fused_repair + (1 - gate_weights) * x
        
        return out

class CWTNet(nn.Module):
    def __init__(self, cdim=48):
        super(CWTNet, self).__init__()
        self.encoder_HV_I = i_hv_encoder()
        self.decoder = Decoder(ch=cdim, out_ch=2, ch_mult=(1, 2, 4), z_channels=cdim*4, is_FirstTrain=False)
        self.VectorQuantizer = VectorQuantizer(n_e=512, c_dim=cdim*4, alpha=1, beta=0)
        self.VectorQuantizer_low = VectorQuantizer(n_e=512, c_dim=cdim*4, alpha=0, beta=1)
        self.Context_Aware_Module = Context_Aware_Module(in_channels=cdim*4)
        self.i_decoder = i_decoder()
        self.trans = RGB_HVI()
        
        self.Gated = nn.Sequential(
            nn.Conv2d(cdim*4, cdim*4, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        hv = hvi[:, 0:2, :, :].to(dtypes)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        i_downsampled = F.avg_pool2d(i, kernel_size=4, stride=4)
        mask = (i_downsampled < x.mean()).float()

        z_q, i_feature, i_jump, intensity = self.encoder_HV_I(i, hv)
        
        # High quantization
        codebook_loss, z_rec_high, _, _, _ = self.VectorQuantizer(z_q)
        
        # Low quantization
        codebook_loss_low, z_rec_low, _, _, _ = self.VectorQuantizer_low(z_q)
        
        z_rec = (1 - mask) * z_rec_high + mask * z_rec_low

        z_rec = self.Context_Aware_Module(z_rec)
        
        hv_rec, code_decoder_output = self.decoder(z_rec)
        i_rec = self.i_decoder(i_feature, code_decoder_output, i_jump, intensity)

        hvi_rec = torch.cat([hv_rec, i_rec], dim=1)
        x_rec = self.trans.PHVIT(hvi_rec)
        
        return x_rec, hv_rec, codebook_loss + codebook_loss_low, i_rec

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi