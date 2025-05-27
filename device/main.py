from screen import Screen

# 加载屏幕并显示加载界面
screen = Screen(0X3C)
screen.disp_loading()
print("loading...")

from gpiozero import Button
from pomodoro_timer import PomodoroTimer
from speech_recognizer import SpeechRecognizer
import threading

if __name__ == "__main__":

    """加载模块"""
    # 计时器  
    model_path = "../model/emotion_net_best.pth"
    timer = PomodoroTimer(model_path, disp=False)  # 非GUI界面运行时，disp 应置为 False
    
    # 语音识别模块
    speech = SpeechRecognizer()
    
    # 按钮
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
    
    """加载结束"""
    print("loading finished.")
        
    """主循环"""
    while True:
        # 屏幕显示
        if (timer.mode == "prepared"):
            screen.disp_prepared()
        elif (timer.mode == "recording"):
            screen.disp_recording()
        elif (timer.mode == "pomodoro"):
            screen.disp_time(timer.remaining_time, True)
        else:
            screen.disp_time(timer.forward_time, False)