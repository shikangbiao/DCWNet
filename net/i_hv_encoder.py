import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.AILO import AILO
from net.CFIGE import CFIGE

class I_Fre_Block(nn.Module):
    def __init__(self, cdim, region_size, kernel_size):
        super(I_Fre_Block, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()
        self.AILO = AILO(in_channels=cdim, out_channels=cdim, region_size=region_size, kernel_size=kernel_size)
        self.CFIGE = CFIGE(dim=cdim)

    def forward(self, x, intensity):
        LL, HL, LH, HH = self.DWT(x)
        HL, HH, LH = self.CFIGE(HL, HH, LH, LL)
        LL = self.AILO(LL, intensity)
        x = self.IWT(LL, HL, LH, HH)
        return x

class i_hv_encoder(nn.Module):
    def __init__(self, 
                 channels=[48, 48, 96, 192],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(i_hv_encoder, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(2, ch1, 3, stride=1, padding=0,bias=False)
        )
        #self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm, Is_Down=True)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm, Is_Down=True)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm, Is_Down=True)
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
        )
        #self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm, Is_Down=True)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm, Is_Down=True)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm, Is_Down=True)
               
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        
        self.trans = RGB_HVI()

        self.I_Fre_Block1 = I_Fre_Block(cdim=ch2, region_size=16, kernel_size=5)
        self.I_Fre_Block2 = I_Fre_Block(cdim=ch3, region_size=8, kernel_size=5)
        self.I_Fre_Block3 = I_Fre_Block(cdim=ch4, region_size=4, kernel_size=5)

    def forward(self, i, hv):
        i_jump = []
        intensity_0 = F.avg_pool2d(i, kernel_size=2, stride=2)
        intensity_1 = F.avg_pool2d(intensity_0, kernel_size=2, stride=2)
        intensity_2 = F.avg_pool2d(intensity_1, kernel_size=2, stride=2)
        intensity = [intensity_0, intensity_1, intensity_2]

        # 浅层
        i_enc0 = self.IE_block0(i)
        hv_0 = self.HVE_block0(hv)

        # 第一个LCA
        i_jump0 = i_enc0
        i_jump.append(i_jump0)

        i_enc1 = self.I_LCA1(i_enc0, hv_0)
        hv_1 = self.HV_LCA1(hv_0, i_enc0)
        i_enc1 = self.I_Fre_Block1(i_enc1, intensity_0)

        # 第二个LCA
        i_jump1 = i_enc1
        i_jump.append(i_jump1)
        i_enc1 = self.IE_block2(i_enc1) # downsample B C H W --> B 2C H/2 W/2
        hv_1 = self.HVE_block2(hv_1)

        i_enc2 = self.I_LCA2(i_enc1, hv_1)
        hv_2 = self.HV_LCA2(hv_1, i_enc1)
        i_enc2 = self.I_Fre_Block2(i_enc2, intensity_1)
        
        # 第三个LCA
        i_jump2 = i_enc2
        i_jump.append(i_jump2)
        i_enc2 = self.IE_block3(i_enc2) # downsample B 2C H/2 W/2 --> B 4C H/4 W/4
        hv_2 = self.HVE_block3(hv_2)

        i_enc3 = self.I_LCA3(i_enc2, hv_2)
        hv_3 = self.HV_LCA3(hv_2, i_enc2)
        i_enc3 = self.I_Fre_Block3(i_enc3, intensity_2)
       
        return hv_3, i_enc3, i_jump, intensity
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
