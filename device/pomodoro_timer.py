import threading
import time
import cv2
from datetime import datetime
from gpiozero import Buzzer
from scorer import Scorer
from onenet import *

class PomodoroTimer:
    def __init__(self, model_path, disp=True):
        # 变量设定
        self.SCORE_TIME = 5    # 每隔多少秒进行一次打分
        self.remaining_time = 0
        self.running = False
        self.forward_time = 0
        self.timer_thread = None
        self.forward_thread = None
        # 加载评分器
        self.scorer = Scorer(model_path)
        # 加载物联网服务
        self.onenet = OneNet()
        # 加载摄像头
        self.frame = None
        self.disp = disp
        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.capture_thread.start()
        # 加载蜂鸣器
        self.buzzer = Buzzer(22)
        # 数据记录
        self.record_scores = []
        self.record_time = 0
        self.record_length = 0
        self.mode = "prepared"
    
    def _capture(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            self.frame = frame
            if self.disp:
                cv2.imshow("test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    def start_pomodoro(self, minutes):
        self.mode = "pomodoro"
        self.record_time = datetime.now().isoformat()
        self.record_length = minutes
        
        self.remaining_time = minutes * 60
        self.running = True
        self.timer_thread = threading.Thread(target=self._countdown, daemon=True)
        self.timer_thread.start()
    
    def _score(self):
        if self.frame is not None:
            score = self.scorer.score(self.frame)
            print(f"SCORE: {score}")
            self.record_scores.append(score)
            if score < 60:
                print("检测到不专注")
                # 蜂鸣器警告
                self.buzzer.on()
                time.sleep(0.5)
                self.buzzer.off()
        
    def _countdown(self):
        while self.remaining_time > 0 and self.running:
            mins, secs = divmod(self.remaining_time, 60)
            print(f"倒计时: {mins:02d}:{secs:02d}")
            time.sleep(1)
            self.remaining_time -= 1
            if self.remaining_time % self.SCORE_TIME == 0:
                # 执行新的线程运行打分，防止阻塞倒计时
                threading.Thread(target=self._score, daemon=True).start()
                
        if self.remaining_time == 0:
            # 记录并上传信息
            self.onenet.report_data(self.record_scores,self.record_time, self.record_length, "Pomodoro")
            print("时间到！")
        
        self.stop()
        
            
    def start_forward(self):
        self.mode = "forward"
        self.record_time = datetime.now().isoformat()
        
        self.forward_time = 0
        self.running = True
        self.forward_thread = threading.Thread(target=self._forward_count, daemon=True)
        self.forward_thread.start()
        
    def _forward_count(self):
        while self.running:
            mins, secs = divmod(self.forward_time, 60)
            print(f"正计时: {mins}:{secs:02d}")
            time.sleep(1)
            if self.forward_time % self.SCORE_TIME == 0:
                # 执行新的线程运行打分，防止阻塞倒计时
                threading.Thread(target=self._score, daemon=True).start()
            self.forward_time += 1
            
        # 记录并上传信息
        self.record_time = self.forward_time
        self.onenet.report_data(self.record_scores,self.record_time, self.record_length, "Forward")
        print("时间到！")  
        self.stop()
         
    def stop(self):
        # 关闭计时
        self.running = False
        self.mode = "prepared"
        self.remaining_time = self.forward_time = 0
        # 蜂鸣器响三声
        for _ in range(3):
            self.buzzer.on()
            time.sleep(0.2)   # 响 0.2 秒
            self.buzzer.off()
            time.sleep(0.2)   # 停 0.2 秒
        # 重置信息
        self.record_scores = []
        self.record_time = 0
        self.record_length = 0
        print("计时已停止。")
        
        
if __name__ == "__main__":    
    model_path = "../model/emotion_net_best.pth"
    
    timer = PomodoroTimer(model_path)
    
    print("开始倒计时 (1分钟) ...")
    
    # timer.start_pomodoro(1)
    timer.start_forward()
    time.sleep(60)
    timer.stop()
    