## FocusManager

https://github.com/Caudalfin24/FocusManager

---

基于树莓派 4B 的沉浸式自习监督助手。

### 项目主要功能

- 信号与多线程的专注计时系统
- 基于深度学习模型的多维度专注检测系统
- vosk 语音识别的驱动功能
- 基于 Onenet 云服务的专注记录后端数据统计
- 环境状态监测提示与调节

### 部署与运行

#### 部署

（可选）创建一个虚拟环境

```
python3 -m venv venv
source venv/bin/activate
```

安装相应依赖
```
pip install -r requirements.txt
```

#### 运行设备

树莓派 OS 在 device/ 文件夹中，执行命令
```
python3 main.py
```

系统进入加载状态，加载结束后，可通过按钮开启计时。会通过语音识别确认专注模式和时间。对应口令为

- “**开始专注**”：正向计时模式
- “**专注X分钟**”：以番茄钟模式倒计时X分钟

专注时间结束后，数据将上报至 Onenet 平台。

#### 运行后端系统

后端服务器在 server/ 文件夹中，运行命令：
```
python3 app.py
```

即可运行后端服务器。浏览器输入如下网址访问后端：
```
https://localhost:5000
```

### 硬件模块与系统接线

- Raspberry Pi 4B
- USB 摄像头
- SSD1306 0.96英寸OLED显示屏
- 按钮模块
- MIC 声音传感器
- 有源蜂鸣器
- 光敏电阻传感器
- RGB LED
- USB迷你麦克风


| 设备        | 通信方式   | 引脚分配                               |
| --------- | ------ | ---------------------------------- |
| 0.96 OLED | I2C    | SDA: GPIO 2, SCL: GPIO 3           |
| 按钮        | 数字输入   | GPIO 17                            |
| 光敏电阻传感器   | 数字输入   | GPIO 4                             |
| RGB LED   | PWM 输出 | R: GPIO 18, G: GPIO 23, B: GPIO 24 |
| MIC 声音传感器 | 数字输入   | GPIO 27                            |
| 有源蜂鸣器     | 数字输出   | GPIO 22                            |
| USB 摄像头   | USB    | USB PORT 0                         |
| USB 麦克风   | USB    | USB PORT 1                         |


### 项目模块结构

#### 项目结构

- `model/`：深度学习模型及训练代码
  - `emotion_net.py`：模型架构
  - `train.py`：模型训练 
- `device/`：物理设备相关代码
  - `main.py`：加载模块与主循环
  - `onenet.py`：Onenet 数据上报相关函数
  - `scorer.py`：专注评分器
  - `pomodoro_timer.py`：计时器
  - `screen.py`：屏幕显示
  - `speech_recognizer.py`：语音识别系统
- `server/`：后端服务
  - `app.py` 主要功能
  - `data.py` 数据获取函数
  - `static/` 静态文件，主要是CSS
  - `templates/` HTML 文件

#### 核心使用库

- **gpiozero**：树莓派 GPIO 控制库
- **pytorch**：情感识别模型构建
- **opencv**：摄像头捕获与图像处理
- **mediapipe**：人脸识别、裁剪与姿态估计
- **paho-mqtt**： 物联网 MQTT 协议，上传至 Onenet
- **luma.oled**：OLED 屏幕绘图
- **vosk**：离线语音识别模型
- **speech_recognition**：语音识别库
- **flask**：服务器系统搭建
- **chart.js**：图表绘制

### 评分算法

![[static/score.drawio.png]]


### 示例演示


![[static/disp.png]]
