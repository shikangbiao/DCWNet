import os
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CWTNet import CWTNet
from measure_niqe_bris import test
from process_HR import process_HR

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
                print(-1)
                input, name = batch[0], batch[1]
            else:
                input, name, path, h, w = batch[0], batch[1], batch[2], batch[3], batch[4]

            if h <= 1024:
                input = input.cuda()
                output, _, _, _ = model(input**gamma)

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True) 
            
                output = torch.clamp(output.cuda(), 0, 1).cuda()
                if not norm_size:
                    output = output[:, :, :h, :w]
                
                output_img = transforms.ToPILImage()(output.squeeze(0))
                output_img.save(output_folder + name[0].split(".")[0] + ".png")
            else:
                process_HR(img_path=path[0], device=device, model=model, output_folder=output_folder, name=name, lednet='lednet', down_factor=8)

        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)

if __name__ == '__main__':
    GPU_ID = "4"
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = False
    num_workers = 1
    alpha = 1
    weight_path1 = ""
    output_folder = './output/NPE/'
    txt = 'unpaired.txt'
    weight_path = "../weights/Unpaired.pth"
    eval_data = DataLoader(dataset=get_unpaired_eval_set("../dataset/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
    eval_net = CWTNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder, norm_size=norm_size, LOL=False, v2=False, unpaired=True, alpha=alpha, gamma=1.0)
    test(output_folder, txt, GPU_ID)
