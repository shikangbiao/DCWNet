import glob
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque
from loss.niqe_utils import *
import os

def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0
        
    for item in tqdm(sorted(glob.glob(im_dir))):
        # print(item)
        im1 = Image.open(item).convert('RGB')
        im1 = np.array(im1)
        if len(im1.shape) != 3 or im1.shape[2] != 3:
            print(f"skip: {item}")
            continue
        score_brisque = brisque.score(im1) 
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)

        avg_brisque += score_brisque
        avg_niqe += score_niqe
        n += 1

        torch.cuda.empty_cache()
    
    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque



def test(base, txt, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    im_dir = base + '/*.png'
    avg_niqe, avg_brisque = metrics(im_dir)
    print("avg_niqe: {:.3f} || avg_brisque: {:.3f}.\n".format(avg_niqe, avg_brisque))

if __name__ == '__main__':
    gpu = "4"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    im_dir = '../output/unpaired/*.png'
    avg_niqe, avg_brisque = metrics(im_dir)
    print("DICM: avg_niqe: {:.3f} || avg_brisque: {:.3f}.\n".format(avg_niqe, avg_brisque))
