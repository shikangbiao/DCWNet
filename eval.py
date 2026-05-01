import os
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CWTNet import CWTNet
import os
from measure import metrics

eval_parser = argparse.ArgumentParser(description='Eval', allow_abbrev=False)
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lolv2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lolv2_syn', action='store_true', help='output lol_v2_syn dataset')
eval_parser.add_argument('--sice', action='store_true', help='output SICE dataset')
eval_parser.add_argument('--sid', action='store_true', help='output sid dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
ep = eval_parser.parse_args()

def eval(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False, alpha=1.0, gamma=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
                    
            input = input.cuda()
            output, _, _, _ = model(input**gamma)
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.cuda(), 0, 1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
        os.mkdir('./output')
    
    norm_size = True
    num_workers = 1
    alpha = None
    gamma = 1.0
    eval_net = CWTNet().cuda()
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("../dataset/LOLv1/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        weight_path = '../weights/LOLv1.pth'

    elif ep.lolv2_real:
        eval_data = DataLoader(dataset=get_eval_set("../dataset/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        weight_path = '../weights/LOLv2real.pth'
        alpha = 0.82
            
    elif ep.lolv2_syn:
        eval_data = DataLoader(dataset=get_eval_set("../dataset/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        weight_path = '../weights/LOLv2syn.pth'
            
    elif ep.sice:
        label_dir = '../dataset/SICE/SICE_Reshape/'
        weight_path = '../weights/SICE.pth'
        norm_size = False
        eval_data_mix = DataLoader(dataset=get_SICE_eval_set("../dataset/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE/SICE_mix/'
        eval(eval_net, eval_data_mix, weight_path, output_folder, 
            norm_size=norm_size, LOL=False, v2=False, unpaired=False, alpha=alpha, gamma=gamma)
        im_dir = output_folder + '*.png'
        avg_psnr_mix, avg_ssim_mix, avg_lpips_mix = metrics(im_dir, label_dir, use_GT_mean=False)

        eval_data_mix = DataLoader(dataset=get_SICE_eval_set("../dataset/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE/SICE_grad/'
        eval(eval_net, eval_data_mix, weight_path, output_folder, 
            norm_size=norm_size, LOL=False, v2=False, unpaired=False, alpha=alpha, gamma=gamma)
        im_dir = output_folder + '*.png'
        avg_psnr_grad, avg_ssim_grad, avg_lpips_grad = metrics(im_dir, label_dir, use_GT_mean=False)
        print("===> Avg.PSNR: {:.3f} dB, Avg.SSIM: {:.3f}, Avg.LPIPS: {:.3f}".format((avg_psnr_mix + avg_psnr_grad) / 2, (avg_ssim_mix + avg_ssim_grad) / 2, (avg_lpips_mix + avg_lpips_grad) / 2))
        exit(0)
    else:
        raise ValueError("--lol, --lolv2_real, --lolv2_syn, --sice")
    eval(eval_net, eval_data, weight_path, output_folder, norm_size=norm_size, LOL=ep.lol, v2=ep.lolv2_real, unpaired=ep.unpaired, alpha=alpha, gamma=gamma)

    use_GT_mean = False
    if ep.lol:
        im_dir = './output/LOLv1/*.png'
        label_dir = '../dataset/LOLv1/eval15/high/'
        use_GT_mean = True
    if ep.lolv2_real:
        im_dir = './output/LOLv2_real/*.png'
        label_dir = '../dataset/LOLv2/Real_captured/Test/Normal/'
    if ep.lolv2_syn:
        im_dir = './output/LOLv2_syn/*.png'
        label_dir = '../dataset/LOLv2/Synthetic/Test/Normal/'

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=use_GT_mean)
    print("===> Avg.PSNR: {:.3f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.3f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.3f} ".format(avg_lpips))
