import requests
from PIL import Image
from io import BytesIO
import json

def get_sm_token():
    with open("./algorithm/config.json", "r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    params = {'username': cfg["username"], 'password': cfg["password"]}
    response = requests.post('https://sm.ms/api/v2/token', params)

    if response.json()['success']:
        print("获取图床token成功")
        return response.json()['data']['token']
    else:
        raise AssertionError("获取图床token失败"+ str(response.json()['message']))

def upload_sm(token, img):
    img = Image.fromarray(img)  
    bytes_io = BytesIO()                # 创建一个BytesIO
    img.save(bytes_io, format='JPEG')   # 写入output_buffer
    headers = {'Authorization': token}
    params = dict()
    params['smfile'] = bytes_io.getvalue()
    response = requests.post('https://sm.ms/api/v2/upload', files=params, headers=headers)
    if response.json()['success']:
        url = response.json()['data']['url']
        return url
    else:
        raise AssertionError("图片上传失败" + str(response.json()['message']))

