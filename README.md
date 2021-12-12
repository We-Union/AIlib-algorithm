# AIlib-algorithm
algorithm for software project


## 相关链接
后端：[AI-lib-BE](https://github.com/We-Union/AI-lib-BE)

前端：[AI-lib-FE](https://github.com/We-Union/AI-lib-FE)

字段说明：

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
|`code`|错误码|
|`output_img_url`|处理得到的图片的url|
|`output_text`|处理得到的文本|
|`msg`|如果错误，展现当前错误的问题|

##

| 错误码 | 说明 |
| --- | --- |
|6001 |输入模型不在已注册模型列表中|
|6002|读取图床图片失败|
|6003|data与模型不匹配|
|6004|param与模型不匹配|
|6005|未检测到符合要求的实体|
|6100|其他未知错误|

> warning:保证CV和NLP的接口函数模块的第一个参数必须是输入数据!!!否则check会出错!!!
