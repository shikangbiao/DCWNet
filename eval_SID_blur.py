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
from process_HR import process_HR
from eval_SID_blur import eval
from measure_SID_blur import test
eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--SID', action='store_true')
eval_parser.add_argument('--Blur', action='store_true')
ep = eval_parser.parse_args()

cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def eval(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False, alpha=1.0, gamma=1.0):
    torch.set_grad_enabled(False)
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        print(alpha)
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

        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)

        _, _, h, w = input.shape
        if h < 1024 and w < 1024:
            with torch.no_grad():
                input = input.cuda()
                output, _, _, _ = model(input**gamma)
            output = torch.clamp(output.cuda(), 0, 1).cuda()
            if not norm_size:
                output = output[:, :, :h, :w]
            output_img = transforms.ToPILImage()(output.squeeze(0))
            output_img.save(output_folder + name[0])
        else:
            process_HR(img=input.cuda(), device='cuda', model=model, output_folder=output_folder, name=name, lednet='lednet', down_factor=8)
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    gpu = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    net = CWTNet().cuda()   
    model_path = f"./SID.pth"
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.eval()
    for index in range(1,230):
        test_dir = "../dataset/Sony_total_dark/test/short/"
        fill_index = '1' + str(index).zfill(4)
        now_dir = test_dir + fill_index + "/"
        SID_folder = './output/SID/'
        if not os.path.exists(SID_folder):          
            os.mkdir(SID_folder)  
        if os.path.exists(now_dir):
            output_folder =  SID_folder + fill_index + "/"
            eval_data = DataLoader(dataset=get_eval_set(now_dir), num_workers=0, batch_size=1, shuffle=False)
            eval(net, eval_data, model_path, output_folder)
    test(0, gpu)
            
