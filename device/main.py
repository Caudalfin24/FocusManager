from screen import Screen

# 加载屏幕并显示加载界面
screen = Screen(0X3C)
screen.disp_loading()
print("loading...")

from gpiozero import Button, DigitalInputDevice, RGBLED
from pomodoro_timer import PomodoroTimer
from speech_recognizer import SpeechRecognizer
import time
import threading

if __name__ == "__main__":

    """加载模块"""
    # 计时器模块
    model_path = "../model/emotion_net_best.pth"
    timer = PomodoroTimer(model_path, disp=False)  # 非GUI界面运行时，disp 应置为 False
    
    # 语音识别模块
    speech = SpeechRecognizer()
    
    # 按钮模块
    button = Button(17,pull_up=False,bounce_time=0.2)
    def on_released():
        if timer.running == False:
            screen.disp_recording()
            timer.mode = "recording"
            mode, time = speech.record()
            print(f"专注模式：{mode}, 时间：{time}")
            if mode == "forward":
                timer_thread = threading.Thread(target=timer.start_forward, daemon=True)
                timer_thread.start()
            else:
                timer_thread = threading.Thread(target=timer.start_pomodoro, args=(time,), daemon=True)
                timer_thread.start()
        else:
            timer.mode = "prepared"
            timer.stop()
        return

    button.when_released = on_released
    
    # 传感器与 LED 模块
    light_sensor = DigitalInputDevice(27)
    light = False
    def on_light_detected():
        light = True
        print("有光（低电平）")

    def on_dark_detected():
        light = False
        print("无光（高电平）")

    light_sensor.when_activated = on_dark_detected
    light_sensor.when_deactivated = on_light_detected
    
    noise_sensor = DigitalInputDevice(4)
    noise = False
    def on_noise_detected():
        noise = True
        print("噪音（低电平）")

    def on_silent_detected():
        noise = False
        print("无噪音（高电平）")

    noise_sensor.when_activated = on_silent_detected
    light_sensor.when_deactivated = on_noise_detected
        
    led = RGBLED(red=18, green=23, blue=24)
    
    last_time = time.time()
    """加载结束"""
    print("loading finished.")
        
    """主循环"""
    while True:
        # 屏幕显示
        if timer.mode == "prepared":
            screen.disp_prepared()
        elif timer.mode == "recording":
            screen.disp_recording()
        elif timer.mode == "pomodoro":
            screen.disp_time(timer.remaining_time, True)
        else:
            screen.disp_time(timer.forward_time, False)
        
        # LED 与传感器控制
        if time.time() - last_time >= 5:
            last_time = time.time()
            timer.onenet.report_light(light)
            timer.onenet.report_noise(noise)
            if light == False:
                led.color=(1,1,1)
            else:
                led.off()