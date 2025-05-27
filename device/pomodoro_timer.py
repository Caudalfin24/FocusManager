import threading
import time
import cv2
from scorer import Scorer

class PomodoroTimer:
    def __init__(self, model_path):
        self.SCORE_TIME = 5    # 每隔多少秒进行一次打分
        self.remaining_time = 0
        self.running = False
        self.forward_time = 0
        self.timer_thread = None
        self.forward_thread = None
        # 加载评分器
        self.scorer = Scorer(model_path)
        # 摄像头
        self.frame = None
        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.capture_thread.start()
        # 数据记录
        self.record_scores = []
        self.record_time = 0
        self.record_length = 0
    
    def _capture(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            self.frame = frame
            cv2.imshow("test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def start_pomodoro(self, minutes):
        self.remaining_time = minutes * 60
        self.running = True
        self.timer_thread = threading.Thread(target=self._countdown, daemon=True)
        self.timer_thread.start()
    
    def _score(self):
        if self.frame is not None:
            score = self.scorer.score(self.frame)
            print(f"SCORE: {score}")
        
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
            # 重置信息
            self.record_scores = []
            self.record_time = 0
            self.record_length = 0
            print("时间到！")
            
    def start_forward(self):
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
        # 重置信息
        self.record_scores = []
        self.record_time = 0
        self.record_length = 0
        print("时间到！")   
    def stop(self):
        self.running = False
        print("计时已停止。")
        
        
if __name__ == "__main__":    
    model_path = "../model/emotion_net_best.pth"
    
    timer = PomodoroTimer(model_path)
    
    print("开始倒计时 (1分钟) ...")
    
    timer.start_pomodoro(1)
    # timer.start_forward()
    time.sleep(60)
    timer.stop()
    