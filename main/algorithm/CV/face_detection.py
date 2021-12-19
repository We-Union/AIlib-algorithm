import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from main.algorithm.CV.utils import resize, show_image

def pad(image, stride=32):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True 

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image

def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep

def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([exp(item) for item in v], v.dtype)
    
    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base
    
    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)

class BBox:
    def __init__(self, label, xyrb, score=0, landmark=None, rotate = False):

        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate
        #避免出现rb小于xy的时候
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = ",".join([str(item[:2]) for item in self.landmark]) if self.landmark is not None else "empty"
        return f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, " + \
            f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def haslandmark(self):
        return self.landmark is not None

    @property
    def xxxxxyyyyy_cat_landmark(self):
        x, y = zip(*self.landmark)
        return x + y

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    @property
    def center(self):
        return [(self.x + self.r) * 0.5, (self.y + self.b) * 0.5]

    # return cx, cy, cx.diff, cy.diff
    def safe_scale_center_and_diff(self, scale, limit_x, limit_y):
        cx = clip_value((self.x + self.r) * 0.5 * scale, limit_x-1)
        cy = clip_value((self.y + self.b) * 0.5 * scale, limit_y-1)
        return [int(cx), int(cy), cx - int(cx), cy - int(cy)]

    def safe_scale_center(self, scale, limit_x, limit_y):
        cx = int(clip_value((self.x + self.r) * 0.5 * scale, limit_x-1))
        cy = int(clip_value((self.y + self.b) * 0.5 * scale, limit_y-1))
        return [cx, cy]

    def clip(self, width, height):
        self.x = clip_value(self.x, width - 1)
        self.y = clip_value(self.y, height - 1)
        self.r = clip_value(self.r, width - 1)
        self.b = clip_value(self.b, height - 1)
        return self

    def iou(self, other):
        return computeIOU(self.box, other.box)

def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
 
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou

def intv(*value):

    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)


def floatv(*value):

    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([float(item) for item in value])
    elif isinstance(value, list):
        return [float(item) for item in value]
    elif value is None:
        return 0
    else:
        return float(value)


def clip_value(value, high, low=0):
    return max(min(value, high), low)


def pad(image, stride=32):

    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True 

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image


def log(v):

    if isinstance(v, tuple) or isinstance(v, list) or isinstance(v, np.ndarray):
        return [log(item) for item in v]
    
    base = np.exp(1)
    if abs(v) < base:
        return v / base
    
    if v > 0:
        return np.log(v)
    else:
        return -np.log(-v)
    
def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([exp(item) for item in v], v.dtype)
    
    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base
    
    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def file_name_no_suffix(path):
    path = path.replace("\\", "/")

    p0 = path.rfind("/") + 1
    p1 = path.rfind(".")

    if p1 == -1:
        p1 = len(path)
    return path[p0:p1]


def file_name(path):
    path = path.replace("\\", "/")
    p0 = path.rfind("/") + 1
    return path[p0:]

"""
    model
"""

class HSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.se(self.pool(x))


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class CBNModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBNModule, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = HSwish()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=2, stride=2,  bias=False):
        super(UpModule, self).__init__()
        self.dconv = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = HSwish()
    
    def forward(self, x):
        x = self.dconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ContextModule(nn.Module):
    def __init__(self, inchannel):
        super(ContextModule, self).__init__()
    
        self.inconv = CBNModule(inchannel, inchannel, 3, 1, padding=1)

        half = inchannel // 2
        self.upconv = CBNModule(half, half, 3, 1, padding=1)
        self.downconv = CBNModule(half, half, 3, 1, padding=1)
        self.downconv2 = CBNModule(half, half, 3, 1, padding=1)

    def forward(self, x):

        x = self.inconv(x)
        up, down = torch.chunk(x, 2, dim=1)
        up = self.upconv(up)
        down = self.downconv(down)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)


class DetectModule(nn.Module):
    def __init__(self, inchannel):
        super(DetectModule, self).__init__()
    
        self.upconv = CBNModule(inchannel, inchannel, 3, 1, padding=1)
        self.context = ContextModule(inchannel)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


class DBFace(nn.Module):
    def __init__(self):
        super(DBFace, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.ReLU(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),           # 0
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),           # 1
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),           # 2
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),   # 3
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 4
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 5
            Block(3, 40, 240, 80, HSwish(), None, 2),                       # 6
            Block(3, 80, 200, 80, HSwish(), None, 1),                       # 7
            Block(3, 80, 184, 80, HSwish(), None, 1),                       # 8
            Block(3, 80, 184, 80, HSwish(), None, 1),                       # 9
            Block(3, 80, 480, 112, HSwish(), SeModule(112), 1),             # 10
            Block(3, 112, 672, 112, HSwish(), SeModule(112), 1),            # 11
            Block(5, 112, 672, 160, HSwish(), SeModule(160), 1),            # 12
            Block(5, 160, 672, 160, HSwish(), SeModule(160), 2),            # 13
            Block(5, 160, 960, 160, HSwish(), SeModule(160), 1),            # 14
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = HSwish()

        self.conv3 = CBNModule(960, 320, kernel_size=1, stride=1, padding=0, bias=False) # 32
        self.conv4 = CBNModule(320, 24, kernel_size=1, stride=1, padding=0, bias=False) # 32
        self.conn0 = CBNModule(24, 24, 1, 1)  # s4
        self.conn1 = CBNModule(40, 24, 1, 1)  # s8
        self.conn3 = CBNModule(160, 24, 1, 1)  # s16

        self.up0 = UpModule(24, 24, 2, 2) # s16
        self.up1 = UpModule(24, 24, 2, 2) # s8
        self.up2 = UpModule(24, 24, 2, 2) # s4
        self.cout = DetectModule(24)
        self.head_hm = nn.Conv2d(48, 1, 1)
        self.head_tlrb = nn.Conv2d(48, 1 * 4, 1)
        self.head_landmark = nn.Conv2d(48, 1 * 10, 1)


    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))

        keep = {"2": None, "5": None, "12": None}
        for index, item in enumerate(self.bneck):
            out = item(out)

            if str(index) in keep:
                keep[str(index)] = out

        out = self.hs2(self.bn2(self.conv2(out)))
        s32 = self.conv3(out)
        s32 = self.conv4(s32)
        s16 = self.up0(s32) + self.conn3(keep["12"])
        s8 = self.up1(s16) + self.conn1(keep["5"])
        s4 = self.up2(s8) + self.conn0(keep["2"])
        out = self.cout(s4)

        hm = self.head_hm(out)
        tlrb = self.head_tlrb(out)
        landmark = self.head_landmark(out)

        sigmoid_hm = hm.sigmoid()
        tlrb = torch.exp(tlrb)
        return sigmoid_hm, tlrb, landmark


    def load(self, file):
        # print(f"load model: {file}")

        if torch.cuda.is_available():
            checkpoint = torch.load(file)
        else:
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint)

def detect(img, model, mean, std, threshold, nms_iou):
    o_img = img
    img = pad(o_img)
    img = ((img / 255.0 - mean) / std).astype(np.float32)
    img = img.transpose(2, 0, 1)

    img_tensor = torch.from_numpy(img)[None]
    model.eval()
    hm, box, landmark = model(img_tensor)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    objs = nms(objs, iou=nms_iou)

    bboxes = [obj.box for obj in objs]
    return bboxes

MEAN = [0.408, 0.447, 0.47]
STD  = [0.289, 0.274, 0.278]

def dnn_detect(img, threshold : float = 0.4, nms_iou : float = 0.5):
    model = DBFace()
    model.load("./model/face.pth")

    bboxes = detect(img, model, MEAN, STD, threshold, nms_iou)
    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (198, 198, 0), 2)
    
    return img


def haar_detect(img, scaleFactor=1.3, minNeighbors=5):
    face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_alt2.xml")
    face_cascade.load('model/haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w,y + h), (198, 198, 0), 2)
    
    return img
    

def detect_face(img, method="dnn", threshold : float = 0.4, nms_iou : float = 0.5, scaleFactor=1.3, minNeighbors=5):
    if method == "dnn":
        result = dnn_detect(img, threshold, nms_iou)
    else:
        result = haar_detect(img, scaleFactor, minNeighbors)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
