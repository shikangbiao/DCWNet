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

class i_decoder(nn.Module):
    def __init__(self, 
                 channels=[48, 48, 96, 192],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(i_decoder, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm, Is_Up=True)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm, Is_Up=True)
        #self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm, Is_Up=True)

        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
        )
        
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()

        self.I_Fre_Block4 = I_Fre_Block(cdim=ch4, region_size=4, kernel_size=5)
        self.I_Fre_Block5 = I_Fre_Block(cdim=ch3, region_size=8, kernel_size=5)
        self.I_Fre_Block6 = I_Fre_Block(cdim=ch2, region_size=16, kernel_size=5)

    def forward(self, i_enc4, hv, i_jump, intensity):
        # 第四个LCA
        i_dec4 = self.I_LCA4(i_enc4, hv[0])
        i_dec4 = self.I_Fre_Block4(i_dec4, intensity[2])

        i_dec3 = self.ID_block3(i_dec4, i_jump[2]) # downsample B 4C H/4 W/4 --> B 2C H/2 W/2

        # 第五个LCA
        i_dec3 = self.I_LCA5(i_dec3, hv[1])
        i_dec3 = self.I_Fre_Block5(i_dec3, intensity[1])
       
        i_dec2 = self.ID_block2(i_dec3, i_jump[1]) # downsample B 2C H/2 W/2 --> B C H W
    
        # 第六个LCA
        i_dec1 = self.I_LCA6(i_dec2, hv[2])
        i_dec1 = self.I_Fre_Block6(i_dec1, intensity[0])

        # 浅层
        i_dec0 = self.ID_block0(i_dec1)
        
        return i_dec0
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
