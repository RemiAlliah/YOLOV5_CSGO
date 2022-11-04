import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = ''
print(device)
device = select_device(device)
print(device)
half = device != 'cpu' #half如果用cuda则是true，else是None

weights = r'D:\yolov5-master\aim-csgo\models\csgo0823.pt' #模型的地址
imgsize = (640,640)
dnn=False
data = 'D:\yolov5-master\data\mydata.yaml'

def load_model():
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsize, s=stride)  # check image size
    if half:
        model.half() #to FP16

    if device != 'cpu':
    #     model(torch.zeros(1, 3, *imgsz)).to(device).type_as(next(model.parameters()))
        model.warmup(imgsz=(1, 3, *imgsz))
    return model
