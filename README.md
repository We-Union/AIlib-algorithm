# AIlib-algorithm
algorithm for software project

[toc]

字段统一一下：

- data: 数据
- model: 模型名称
- param: 参数



## 算法入口函数说明

Mode：

```python
def interface(data : str, model : str, param : dict):
    """
    	data  : 数据（如果是多个数据，逗号分隔）
    	model : 调用的模型名称
    	param : 参数
    """
    ...
```

## main函数返回字段说明
|字段|说明|
| :---: |  :---: |
|`code`|0:成功 1:输入模型不在已注册模型列表中 2:读取图床图片失败 3:data与模型不匹配 4:param与模型不匹配 5:未检测到符合要求的实体 6:其他的错误|
|`output_img_url`|处理得到的图片的url|
|`output_text`|处理得到的文本|
|`msg`|如果错误，展现当前错误的问题|


> warning:保证CV和NLP的接口函数模块的第一个参数必须是输入数据!!!否则check会出错!!!