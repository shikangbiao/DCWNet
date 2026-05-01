from net.HVI_transform import RGB_HVI
from net.encoder_decoder import Encoder, Decoder
import torch
import torch.nn as nn
from net.transformer_utils import *

def add_mixed_noise_simple(hv_tensor, gauss_std, poisson_scale):
    device = hv_tensor.device
    dtype = hv_tensor.dtype
    
    hv_cpu = hv_tensor.cpu().float()
    
    original_min = hv_cpu.min()
    original_max = hv_cpu.max()
    original_range = original_max - original_min
    
    if original_range > 0:
        hv_cpu = (hv_cpu - original_min) / (original_range + 1e-8)
    else:
        hv_cpu = hv_cpu
    
    scaled = poisson_scale * hv_cpu
    poisson_noise = torch.poisson(scaled) / poisson_scale - hv_cpu
    gaussian_noise = torch.randn_like(hv_cpu) * gauss_std
    
    noisy_hv = hv_cpu + poisson_noise + gaussian_noise
    
    noisy_hv = torch.clamp(noisy_hv, 0, 1)
    
    if original_range > 0:
        noisy_hv = noisy_hv * original_range + original_min
    else:
        noisy_hv = noisy_hv
    
    return noisy_hv.to(device=device, dtype=dtype)

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, c_dim, alpha, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.c_dim = c_dim
        self.alpha = alpha
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.c_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.c_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - \
                2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).cuda()
        # to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = self.alpha * torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    
    def get_latent(self, min_encoding_indices, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        min_encoding_indices = min_encoding_indices.view(-1)
        min_encoding_indices = min_encoding_indices.unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).cuda()
        # to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

class VQGAN(nn.Module):
    def __init__(self, cdim=48):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(ch=cdim, in_channels=2, z_channels=cdim*4)
        self.decoder = Decoder(ch=cdim, out_ch=2, z_channels=cdim*4, is_FirstTrain=True)
        self.VectorQuantizer = VectorQuantizer(n_e=512, c_dim=cdim*4, alpha=1, beta=1)
        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        hv = hvi[:, 0:2, :, :].to(dtypes)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        with torch.no_grad():
            noisy_hv = add_mixed_noise_simple(hv, gauss_std=0.001, poisson_scale=100.0)

        z_q = self.encoder(noisy_hv)
        codebook_loss, z_rec, _, _, _ = self.VectorQuantizer(z_q)
        P_h, hv_rec, _ = self.decoder(z_rec)
        
        hvi_rec = torch.cat([hv_rec, i], dim=1)
        x_rec = self.trans.PHVIT(hvi_rec)
    
        return x_rec, hv_rec, P_h, codebook_loss
    
    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi