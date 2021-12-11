import numpy as np
import cv2 as cv

import sys
from pprint import pprint

for path in sys.path:
    if path[0] in ['e', 'E']:
        print(path)

from main.my_io import get_sm_token, upload_sm
from main.algorithm.CV import transform_to_painting, url_imread, show_image
from main.algorithm.NLP import *
from main.check import check_func_params


register_cv_algorithm = {
    "transform_to_painting" : transform_to_painting 
}

register_multi_cv_algorithm = {

}

register_nlp_algorithm = []

def main(data : str=None, model : str=None, param : dict=None):
    # init
    global_token = get_sm_token()

    # check if cv or nlp
    if model in register_cv_algorithm:
        img_url = data
        try:
            img = url_imread(img_url)
        except:
            return {
                "code" : 2,
                "output_img_url" : None,
                "output_text" : None,
                "msg":"读取图床图片失败",
            }

        if not check_func_params(register_cv_algorithm[model], param):
            return {
                "code" : 4,
                "output_img_url" : None,
                "output_text" : None,
                "msg":"参数与模型不匹配",
            }
        
        output_image = register_cv_algorithm[model](img, **param)
        url = upload_sm(global_token, output_image)

        return {
            "code" : 0,
            "output_img_url" : url,
            "output_text" : ""
        }

    elif model in register_multi_cv_algorithm:
        img_urls = data.split(",")
    elif model in register_nlp_algorithm:
        ...
    else:
        return {
            "code" : 1,
            "output_img_url" : None,
            "output_text" : None,
            "msg":"输入模型不在已注册模型列表",
        }

main()