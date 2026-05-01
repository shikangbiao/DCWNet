import os
import cv2
import torch
from torchvision.transforms.functional import normalize
import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
from torch.nn import functional as F
from torchvision.transforms.functional import crop

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]

def image_to_patches(image, psize, stride):
        psize_h, psize_w = psize if isinstance(psize, tuple) else (psize, psize)
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)

        h, w = image.shape[-2:]
        h_list = [i for i in range(0, h - psize_h + 1, stride_h)]
        w_list = [i for i in range(0, w - psize_w + 1, stride_w)]
        corners = [(hi, wi) for hi in h_list for wi in w_list]
        patches = []
        for hi in h_list:
            for wi in w_list:
                patches.append(crop(image, hi, wi, psize_h, psize_w))
        patches = torch.stack(patches)
        return patches, h_list, w_list

def patches_to_image(patches,  h_list, w_list, psize, stride, H, W, device):
    psize_h, psize_w = psize if isinstance(psize, tuple) else (psize, psize)
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    overlap = psize_h - stride_h

    restore = torch.zeros((3, H, W)).to(device)  
    patch_row = torch.zeros((3, psize_h, W)).to(device)  # row
    # patch的overlap的weight 0-->overlap
    weight1 = torch.arange(0, overlap).view(1, -1) / (overlap-1)
    weight1 = torch.repeat_interleave(weight1, psize_h, dim=0)
    weight1 = weight1.repeat(3, 1, 1).to(device)

    # patch_row中overlap的weight  overlap-->0
    weight2 = torch.flip(weight1, dims=[2]).to(device)

    # patch_row的weight
    weight3 = torch.arange(0, overlap).view(-1, 1) / (overlap - 1)
    weight3 = torch.repeat_interleave(weight3, W, dim=1)
    weight3 = weight3.repeat(3, 1, 1).to(device)

    # resore image的weight
    weight4 = torch.flip(weight3, dims=[1]).to(device)

    for hi in h_list:
        torch.zero_(patch_row)  # patch
        for wi in w_list:
            patch = patches.pop(0)
            # print("wi:", wi)
            # print(patch.shape)
            if wi == 0:
                patch_row[:, :, wi:wi+psize_w] += patch
            else:

                patch_row[:, :, wi:wi+overlap] = patch_row[:, :, wi: wi+overlap] * weight2
                patch[:, :, 0:overlap] = patch[:, :, 0: overlap] * weight1
                patch_row[:, :, wi:wi+psize_w] += patch
        if hi == 0:
            restore[:, hi:hi+psize_h, :] += patch_row
        else:
            restore[:, hi:hi+overlap, :] = restore[:, hi:hi+overlap, :] * weight4
            patch_row[:, 0:overlap, :] = patch_row[:, 0:overlap, :] * weight3
            restore[:, hi:hi+psize_h, :] += patch_row

    return restore

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def process_HR(img_path, device, model, output_folder, name, lednet='lednet', down_factor = 8):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # prepare data
    img_t = img2tensor(img / 255., bgr2rgb=True, float32=True).to(device)

    if not lednet == 'lednet':
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        img_t = check_image_size(img_t, down_factor)  

        patch_size = 768
        stride = 384
        mul = patch_size
        h, w = img_t.shape[2], img_t.shape[3]
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        input_ = F.pad(img_t, (0, padw, 0, padh), 'reflect')
            
        patches, h_list, w_list = image_to_patches(input_[0], patch_size, stride)
        restored_patches = []
        for patch in patches.split(1):
            restore, _, _, _ = model(patch)
            restored_patches.extend(restore)
    
        restored = patches_to_image(restored_patches, h_list, w_list, patch_size, stride, H, W, device)
        restored = restored.unsqueeze(0)

        # restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :h, :w]
        pred = restored
        if isinstance(pred, list):
            pred = pred[-1]
        output_t = pred

        # output_t = output_t[:,:,:H,:W]

        if lednet == 'lednet':
            output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))
        else:
            output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        del output_t
        output = output.astype('uint8')
        # save restored img
        save_restore_path = output_folder + name[0].split(".")[0] + ".png"
        imwrite(output, save_restore_path)

def process_HR1(img, device, model, output_folder, name, lednet='lednet', down_factor = 8):
    if not lednet == 'lednet':
        normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

    with torch.no_grad():
        img_t = check_image_size(img, down_factor) 
        patch_size = 768
        stride = 384
        mul = patch_size
        h, w = img_t.shape[2], img_t.shape[3]
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        input_ = F.pad(img_t, (0, padw, 0, padh), 'reflect')
            
        patches, h_list, w_list = image_to_patches(input_[0], patch_size, stride)
        restored_patches = []
        for patch in patches.split(1):
            restore, _, _, _ = model(patch)
            restored_patches.extend(restore)
    
        restored = patches_to_image(restored_patches, h_list, w_list, patch_size, stride, H, W, device)
        restored = restored.unsqueeze(0)

        # restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :h, :w]
        pred = restored
        if isinstance(pred, list):
            pred = pred[-1]
        output_t = pred

        # output_t = output_t[:,:,:H,:W]

        if lednet == 'lednet':
            output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))
        else:
            output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        del output_t
        output = output.astype('uint8')
        # save restored img
        save_restore_path = output_folder + name[0].split(".")[0] + ".png"
        imwrite(output, save_restore_path)
