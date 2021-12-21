import inspect
from main.algorithm.CV import transform_to_painting

def check_func_params(f, param : dict) -> bool:
    f_parser = inspect.getargspec(f)
    args = f_parser.args[1:]
    return set(args) == set(param.keys())

def check_return_code(code):
    if code == 6001:
        return {
            "code": 6001,
            "msg": "输入模型不在已注册模型列表中",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6002:
        return {
            "code": 6002,
            "msg": "读取图床图片失败",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6003:
        return {
            "code": 6003,
            "msg": "data与模型不匹配",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6004:
        return {
            "code": 6004,
            "msg": "param与模型不匹配",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6005:
        return {
            "code": 6005,
            "msg": "未检测到符合要求的实体",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6006:
        return {
            "code" : 6006,
            "msg": "输入参数不在范围内",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6007:
        return {
            "code" : 6007,
            "msg": "服务器内存占用溢出",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6008:
        return {
            "code" : 6008,
            "msg": "服务器内存占用溢出",
            "output_img_url": "",
            "output_text": "",
        }
    elif code == 6009:
        return {
            "code" : 6009,
            "msg": "输入文本为空",
            "output_img_url": "",
            "output_text": "",
        }

    elif code == 6010:
        return {
            "code" : 6010,
            "msg": "超出支持语言",
            "output_img_url": "",
            "output_text": "",
        }

    elif code == 6011:
        return {
            "code" : 6011,
            "msg": "词云文本不存在有效语义信息",
            "output_img_url": "",
            "output_text": "",
        }
    
    elif code == 6012:
        return {
            "code" : 6012,
            "msg": "词向量可视化输入单词都不在表中，试试两个字的名词",
            "output_img_url": "",
            "output_text": "",
        }

if __name__ == "__main__":
    ...