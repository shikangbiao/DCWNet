
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg", ".PNG", ".JPEG", ".BMP"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
