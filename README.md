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
|6006|输入参数不在范围内|
|6007|服务器内存占用溢出|
|6008|服务器显存占用溢出|
|6009|输入文本为空|
|6100|其他未知错误|

### CV模型参数解释

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

#### 4. reconstruct （高分辨率重建）

输入：`img`，代表一张图片。

> 在使用前，请将我给的模型文件`reconstruct.pth`放在根目录的model文件夹下

| 参数     | 类型值 | 范围\可选值   | 默认值 | 备注                                                         |
| -------- | ------ | ------------- | ------ | ------------------------------------------------------------ |
| device   | string | "cuda", "cpu" | 自适应 | 运算设备                                                     |
| scale    | int    | > 2          | 4      | 放大倍率                                                     |
| outscale | float  | > 0           | 0   | 输出图像和输入图像在尺寸上的倍数关系，若为0，则`outscale==scale` |



#### 5. stitching（图像拼接）

输入：`imgs`，一个两张图片组成的列表，`imgs==[img1, img2]`

| 参数           | 类型值 | 范围\可选值                            | 默认值 | 备注             |
| -------------- | ------ | -------------------------------------- | ------ | ---------------- |
| feature        | string | "akaze", "kaze", "mser", "orb", "sift" | "sift" | 关键点特征       |
| ratio          | float  | > 0                                    | 0.75   | 距离抑制值       |
| reproThreshold | float  | > 0                                    | 0.4    | 仿射变换投影阈值 |



#### 6. detect_face（人脸检测）

输入：`img`，代表一张图片。

> 在使用前，请将我给的模型文件`haarcascade_frontalface_alt2.xml`放在根目录的model文件夹下

| 参数         | 类型值 | 范围\可选值   | 默认值 | 备注                     |
| ------------ | ------ | ------------- | ------ | ------------------------ |
| method       | string | "dnn", "haar" | "dnn"  | 人脸检测算法             |
| threshold    | float  | > 0           | 0.4    | bbox得分阈值             |
| nms_iou      | float  | > 0           | 0.5    | 非极大值抑制采用的IOU    |
| scaleFactor  | float  | > 0           | 1.3    | haar人脸检测缩放因子     |
| minNeighbors | int    | > 0           | 5      | haar保留预选框的bbox数量 |



#### 7. ocr_print & ocr_val（打印体识别和验证码识别）

输入：`img`，代表一张图片。

> 使用前，请根据文档最后的“特殊依赖库1.1”安装相关库

这两个函数都没有参数，需要注意，它们返回的是字符串。



#### 8. equalize_hist（图像均衡化）

输入：`img`，代表一张图片。

| 参数         | 类型值 | 范围\可选值 | 默认值 | 备注                                            |
| ------------ | ------ | ----------- | ------ | ----------------------------------------------- |
| local        | bool   | true/false  | false  | 是否使用局部均衡化                              |
| clipLimit    | float  | > 0         | 4.0    | `local==True`才有效。局部均衡化裁剪阈值         |
| tileGridSize | int    | > 0         | 4      | `local==True`才有效。局部均衡化裁剪阈值格窗大小 |



#### 9. OSTU_split（大津阈值法）

输入：`img`，代表一张图片。

| 参数      | 类型值 | 范围\可选值       | 默认值 | 备注                             |
| --------- | ------ | ----------------- | ------ | -------------------------------- |
| blur_size | int    | [3, 5, 7, 9, ...] | 3      | 模糊窗口大小                     |
| blur_std  | float  | > 0               | 1      | 模糊方差                         |
| reverse   | bool   | true/false        | false  | 是否对最终的二值图像进行像素反转 |





---

### NLP模型参数解释

#### 1. kanji_cut（中文分词）

输入：`text`，代表一句话。

| 参数    | 类型值 | 范围\可选值 | 默认值 | 备注             |
| ------- | ------ | ----------- | ------ | ---------------- |
| spliter | string | any         | " "    | 输出结果的分隔符 |








> warning:保证CV和NLP的接口函数模块的第一个参数必须是输入数据!!!否则check会出错!!!

---

### 特殊依赖库

#### 特殊依赖库1.1

[老妖Terry/muggle-ocr - 码云 - 开源中国 (gitee.com)](https://gitee.com/ruifeng96150/muggle-ocr?_from=gitee_search)

访问上述链接，下载安装包，然后

```bash
$pip install muggle-ocr
```

