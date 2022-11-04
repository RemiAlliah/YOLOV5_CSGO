from grabscreen import grab_screen
from cs_model import load_model
import cv2
import win32gui
import win32con
import torch
import numpy as np
from utils.general import non_max_suppression
from utils.general import check_img_size
from utils.augmentations import letterbox
from utils.general import scale_coords
from utils.general import xyxy2xywh
# from utils.torch_utils import select_device

import pynput #控制鼠标移动
from mouse_control import lock

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
imgsz=(640, 640)

conf_thres = 0.4
iou_thres = 0.05
classes = None
agnostic_nms = False
max_det = 1000
x, y = (2560, 1440)
re_x, re_y = (2560, 1440)

model = load_model()
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)

lock_mode = False
mouse = pynput.mouse.Controller()

with pynput.mouse.Events() as events:
    while True:
        it = next(events)
        while it is not None and not isinstance(it, pynput.mouse.Events.Click):
            it = next(events)
        if it is not None and it.button == it.button.x2 and it.pressed:
            lock_mode = not lock_mode
            print('lock mode','on' if lock_mode else 'off')
# while True:
#
        img0 = grab_screen(region=(0, 0, x, y))
        img0 = cv2.resize(img0, (re_x, re_y))
        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device) #对图片进行格式的转换
        img = img.half() if half else img.float()# uint8 to fp16/32
        img /= 255 #归一化操作，
        if len(img.shape) == 3:
            img = img[None] # img = img.unsqueeze(0)

        #Inference
        pred = model(img, augment=False, visualize=False)
        #NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print(pred)
        aims = []
        #Process predictions
        for i, det in enumerate(pred):  # per image
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #bbox:(tag, x_center, y_center, x_width, y_width)
                    """
                    tag:{0:ct_head,1:ct_body,2:t_head,3:t_body}
                    """
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    aim = ('%g ' * len(line)).rstrip() % line
                    # print(aim)
                    # print(type(aim))
                    aim = aim.split(' ')
                    # print(aim)
                    aims.append(aim)

                #如果框框不是空的，就标出来
                if len(aims):
                    if lock_mode:
                        lock(aims, mouse, x, y)
                    for i, det in enumerate(aims):
                        _, x_center, y_center, width, height = det
                        x_center, width = re_x * float(x_center), re_x * float(width)
                        y_center, height = re_y * float(y_center), re_y * float(height)
                        top_left = (int(x_center - width / 2.), int(y_center - height / 2.)) #cv2库不支持传入浮点数，所以要转成int
                        bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                        color = (0, 255, 0)#绿色的框
                        cv2.rectangle(img0, top_left, bottom_right, color, 3)#3表示线条的粗细





        cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('csgo-detect', re_x // 3, re_y // 3)
        cv2.imshow('csgo-detect', img0)

        hwnd = win32gui.FindWindow(None, 'csgo-detect')
        CVRECT = cv2.getWindowImageRect('csgo-detect')
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # a = grab_screen(region=(0,0,2560,1440))
        #
        # cv2.imshow('1',a)1q
        # cv2.waitKey(0)