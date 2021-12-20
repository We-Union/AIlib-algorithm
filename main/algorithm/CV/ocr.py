import muggle_ocr
import cv2 as cv
import os

from torch.nn.modules.module import register_module_backward_hook

temp_file = "temp.jpg"

def ocr_val(img):
    sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.Captcha)
    cv.imwrite(temp_file, img)
    with open(temp_file, "rb") as f:
        ocr_bytes = f.read()
    text = sdk.predict(ocr_bytes)
    os.remove(temp_file)
    return None,text

def ocr_print(img):
    sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.OCR)
    cv.imwrite(temp_file, img)
    with open(temp_file, "rb") as f:
        ocr_bytes = f.read()
    text = sdk.predict(ocr_bytes)
    os.remove(temp_file)

    return None,text