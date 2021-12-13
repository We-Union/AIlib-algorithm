# AIlib-algorithm
algorithm for software project


## 相关链接
后端：[AI-lib-BE](https://github.com/We-Union/AI-lib-BE)

前端：[AI-lib-FE](https://github.com/We-Union/AI-lib-FE)


---

## 算法入口函数说明

字段说明：

- data: 数据
- model: 模型名称
- param: 参数

算法与后端的交互逻辑定义在`interface.py`中：

```python
def main(data : str, model : str, param : dict):
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

### 错误码

| 错误码 | 说明 |
| --- | --- |
|6001 |输入模型不在已注册模型列表中|
|6002|读取图床图片失败|
|6003|data与模型不匹配|
|6004|param与模型不匹配|
|6005|未检测到符合要求的实体|
|6006||
|6100|其他未知错误|

### 模型参数解释

#### 1. transform_to_painting（手绘风格转换）

输入：`img`，代表一张图片。

| 参数 | 类型值 | 范围\可选值 |默认值|备注|
| --- | --- | --- | --- | --- |
|depth|int|[0,255]|100| 梯度增益系数   |
|blur|bool|true/false|false|是否进行预模糊|
|blur_size|int|[3, 5, 7, 9, ...]|3|模糊窗口大小|
|blur_std|float|> 0|1|模糊方差|
|denoise|bool|true/false|false|是否进行去噪|
|denoise_size|int|[3, 5, 7, 9, ...]|3|去噪窗口大小|

#### 2. scanning（文档扫描）

输入：`img`，代表一张图片。

| 参数   | 类型值 | 范围\可选值 | 默认值 | 备注         |
| ------ | ------ | ----------- | ------ | ------------ |
| height | int    | > 0         | 500    | 输出图片大小 |

#### 3. sift_matching（图片匹配）

输入：`imgs`，一个两张图片组成的列表，`imgs==[img1, img2]`

| 参数       | 类型值 | 范围\可选值                            | 默认值  | 备注                                         |
| ---------- | ------ | -------------------------------------- | ------- | -------------------------------------------- |
| feature    | string | "akaze", "kaze", "mser", "orb", "sift" | "sift"  | 关键点特征                                   |
| match_rule | string | "brute", "knn"                         | "brute" | 关键点匹配算法                               |
| k          | int    | > 0                                    | 3       | knn匹配算法的k值                             |
| show_lines | int    | > 0                                    | 30      | 最终展示的匹配图中的连接对应关键点的连线数量 |

#### 4.







> warning:保证CV和NLP的接口函数模块的第一个参数必须是输入数据!!!否则check会出错!!!
