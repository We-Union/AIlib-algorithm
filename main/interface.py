import numpy as np
import cv2 as cv

import sys
from pprint import pprint

# for path in sys.path:
#     if path[0] in ['e', 'E']:
#         print(path)

from main.my_io import get_sm_token, upload_sm
from main.algorithm.CV import transform_to_painting, url_imread, show_image
from main.algorithm.CV import scanning, sift_matching, reconstruct
from main.algorithm.NLP import *
from main.check import check_func_params, check_return_code

register_cv_algorithm = {
    "transform_to_painting": transform_to_painting,
    "scanning": scanning,
    "reconstruct": reconstruct
}

register_multi_cv_algorithm = {
    "sift_matching": sift_matching
}

register_nlp_algorithm = []


def main(data: str = None, model: str = None, param: dict = None):
    # init
    global_token = get_sm_token()

    # check if cv or nlp
    if model in register_cv_algorithm:
        img_url = data
        try:
            img = url_imread(img_url)
        except:
            return check_return_code(6002)

        if not check_func_params(register_cv_algorithm[model], param):
            return check_return_code(6004)

        output_image = register_cv_algorithm[model](img, **param)

        if isinstance(output_image, int):
            err_code = output_image
            return check_return_code(err_code)

        url = upload_sm(global_token, output_image)

        return {
            "code": 0,
            "msg": "",
            "output_img_url": url,
            "output_text": "分析成功"
        }

    elif model in register_multi_cv_algorithm:
        img_urls = data.split(",")
        try:
            img_list = [url_imread(url) for url in img_urls]
        except:
            return check_return_code(6002)

        if not check_func_params(register_cv_algorithm[model], param):
            return check_return_code(6004)

        output_image = register_cv_algorithm[model](img_list, **param)
        url = upload_sm(global_token, output_image)

        return {
            "code": 0,
            "msg": "",
            "output_img_url": url,
            "output_text": "分析成功"
        }

    elif model in register_nlp_algorithm:
        ...
    else:
        return check_return_code(6001)


if __name__ == "__main__":
    main()
