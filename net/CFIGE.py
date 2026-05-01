import torch
import torch.nn as nn
from net.transformer_utils import *

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0) # B

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False) # B, C, 1

    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)


    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )

    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )
    return filtered_candidate_maps

def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, C, H*W)
    candidate_maps: (B, C, H*W)
    """
    distances = torch.cdist(input_maps, candidate_maps) # B,C,C

    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

class Matching(nn.Module):
    def __init__(self, dim=32, match_factor=1):
        super(Matching, self).__init__()
        self.num_matching = int(dim/match_factor)
    def forward(self, x, perception):
        b, c, h, w = x.size()
        x = x.flatten(2, 3)
        perception = perception.flatten(2, 3)
        filtered_candidate_maps = neirest_neighbores_on_l2(x, perception, self.num_matching)
        filtered_candidate_maps = filtered_candidate_maps.reshape(b, self.num_matching, h, w)
        return filtered_candidate_maps

class Matching_transformation(nn.Module):
    def __init__(self, dim=32, match_factor=1):
        super(Matching_transformation, self).__init__()
        self.matching = Matching(dim=dim, match_factor=match_factor)

    def forward(self, x, perception):
        filtered_candidate_maps = self.matching(x, perception)
        out = filtered_candidate_maps * x

        return out

class CFIGE(nn.Module):
    def __init__(self, dim):
        super(CFIGE, self).__init__()
        self.Matching_transformation_1 = Matching_transformation(dim=dim)
        self.Matching_transformation_2 = Matching_transformation(dim=dim)
        self.Matching_transformation_3 = Matching_transformation(dim=dim)
        
        self.pconv = nn.Conv2d(in_channels=3 * dim, out_channels=2 * dim, kernel_size=1)
        self.act = nn.Sigmoid()
        
        self.high_freq_conv_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.high_freq_conv_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.high_freq_conv_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
    def forward(self, HL, HH, LH, LL):
        res_HL = HL.clone()
        res_HH = HH.clone()
        res_LH = LH.clone()
        HL = self.Matching_transformation_1(HL, LL)
        HH = self.Matching_transformation_2(HH, LL)
        LH = self.Matching_transformation_3(LH, LL)
        W = torch.cat([HL, HH, LH], dim=1)

        W = self.act(self.pconv(W))

        W1, W2 = torch.chunk(W, 2, dim=1)
        HH = HH + W1 * HL + W2 * LH

        return self.high_freq_conv_1(HL) + res_HL, self.high_freq_conv_2(HH) + res_HH, self.high_freq_conv_3(LH) + res_LH