from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont
import time

class Screen:
    def __init__(self, address=0x3C):
        self.serial = i2c(port=1, address=address)  # 确保这个地址与你的i2cdetect一致
        self.device = ssd1306(self.serial, width=128, height=64)
        self.device.clear()
        self.font = ImageFont.load_default()
    
    def disp_loading(self):
        """显示加载图像"""
        # 创建空图像
        image = Image.new("1", self.device.size)
        draw = ImageDraw.Draw(image)

        # 居中显示 "LOADING"
        text = "LOADING" + "..."
        w, h = draw.textsize(text, font=self.font)
        x = (self.device.width - w) // 2
        y = (self.device.height - h) // 2

        draw.text((x, y), text, font=self.font, fill=255)
        self.device.display(image)

    def disp_prepared(self):
        """显示就绪图像（番茄）"""
        # 创建黑白图像
        image = Image.new("1", self.device.size)
        draw = ImageDraw.Draw(image)

        # --- 绘制番茄图案 ---
        # 圆形主体（番茄）
        tomato_radius = 20
        center_x = self.device.width // 2
        center_y = self.device.height // 2 - 10  # 稍微往上偏移

        draw.ellipse(
            (center_x - tomato_radius, center_y - tomato_radius,
            center_x + tomato_radius, center_y + tomato_radius),
            outline=255,
            fill=0
        )

        # 简单绘制番茄顶部的叶子（用几条线模拟）
        leaf_length = 8
        for angle in [0, 72, 144, 216, 288]:
            x2 = center_x + leaf_length * __import__('math').cos(__import__('math').radians(angle))
            y2 = center_y - tomato_radius + leaf_length * __import__('math').sin(__import__('math').radians(angle))
            draw.line((center_x, center_y - tomato_radius, x2, y2), fill=255)

        # --- 添加文字 ---
        font = ImageFont.load_default()
        text = "PRESS TO START"
        w, h = draw.textsize(text, font)
        draw.text(((self.device.width - w) // 2, self.device.height - h - 2), text, font=font, fill=255)

        # 显示图像
        self.device.display(image)
    
    def disp_recording(self):
        """显示录音中图像"""
        image = Image.new("1", self.device.size)
        draw = ImageDraw.Draw(image)

        # --- 绘制麦克风图标 ---
        center_x = self.device.width // 2
        top_y = 10

        mic_width = 16
        mic_height = 24
        mic_left = center_x - mic_width // 2
        mic_right = center_x + mic_width // 2
        mic_bottom = top_y + mic_height

        # 麦克风主体（圆角矩形）
        draw.rectangle([mic_left, top_y + 4, mic_right, mic_bottom], outline=255, fill=0)
        draw.ellipse([mic_left, top_y, mic_right, top_y + 8], outline=255, fill=0)  # 顶部圆弧

        # 麦克风底座
        draw.line([(center_x, mic_bottom), (center_x, mic_bottom + 6)], fill=255)
        draw.line([(center_x - 6, mic_bottom + 6), (center_x + 6, mic_bottom + 6)], fill=255)

        # --- 显示 RECORDING 文字 ---
        font = ImageFont.load_default()
        text = "RECORDING"
        text_w, text_h = draw.textsize(text, font)
        draw.text(((self.device.width - text_w) // 2, self.device.height - text_h - 2), text, font=font, fill=255)

        # 显示内容
        self.device.display(image)

    def disp_time(self, seconds=0, pomodoro=True):
        """给定时间（秒数），显示时间图像"""
        # 创建黑白图像
        image = Image.new("1", self.device.size)
        draw = ImageDraw.Draw(image)

        # --- 1. 画时钟图标（左侧） ---
        clock_x = 10
        clock_y = 10
        radius = 10

        # 外圆
        draw.ellipse((clock_x - radius, clock_y - radius, clock_x + radius, clock_y + radius), outline=255, fill=0)
        # 时针（12点方向）
        draw.line((clock_x, clock_y, clock_x, clock_y - 6), fill=255)
        # 分针（3点方向）
        draw.line((clock_x, clock_y, clock_x + 5, clock_y), fill=255)

        # --- 2. 显示时间 ---
        minutes = seconds // 60
        secs = seconds % 60
        time_text = f"{minutes:02}:{secs:02}"

        try:
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            large_font = ImageFont.load_default()

        text_w, text_h = draw.textsize(time_text, font=large_font)
        time_x = 40  # 稍右偏移，避开左侧时钟
        time_y = (self.device.height - text_h) // 2 - 8

        draw.text((time_x, time_y), time_text, font=large_font, fill=255)

        # --- 3. 显示小字说明 ---
        mode_text = "POMODORO" if pomodoro else "FORWARD"
        small_font = ImageFont.load_default()
        mode_w, mode_h = draw.textsize(mode_text, font=small_font)
        draw.text((time_x, time_y + text_h + 2), mode_text, font=small_font, fill=255)

        # 显示到屏幕
        self.device.display(image)

if __name__ == "__main__":
    screen = Screen()
    seconds = 40
    while seconds >= 0:
        screen.disp_time(seconds,False)
        seconds -= 1
        time.sleep(1)

    # screen.disp_recording()
    # time.sleep(5)
    # screen.disp_loading()
    # time.sleep(5)