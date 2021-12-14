# refinement : LSTM-Kirigaya
# reference : https://github.com/xinntao/Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from main.algorithm.CV.utils import resize
import cv2 as cv
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reconstruct(img, device=DEVICE, scale=4, outscale=None):
    if scale < 3:
        return 6006
    model_state_dict = torch.load("model/reconstruct.pth")
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=scale
    )

    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    h, w = img.shape[0], img.shape[1]

    is_gray = bool(len(img.shape) == 2)
    
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB if is_gray else cv.COLOR_BGR2RGB)
    img = img / 255
    img = torch.FloatTensor(img.transpose((2, 0, 1))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        try:
            output = model(img)
        except:
            if device == "cuda":
                return 6008
            elif device == "cpu":
                return 6007

    out_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    out_img = out_img[[2, 1, 0], :, :].transpose((1, 2, 0))

    if is_gray:
        out_img = cv.cvtColor(out_img, cv.COLOR_BGR2GRAY)
    
    out_img = out_img * 255
    out_img = out_img.round().astype('uint8')

    if outscale is not None:
        out_img = resize(out_img, height=int(h * outscale))
    
    return out_img