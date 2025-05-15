import threading
import time
import cv2
from scorer import Scorer

pic_path = "../model/data/FER2013/test/angry/PrivateTest_12766285.jpg"
pic = cv2.imread(pic_path)

class PomodoroTimer:
    def __init__(self, model_path):
        self.remaining_time = 0
        self.running = False
        self.forward_time = 0
        self.timer_thread = None
        self.forward_thread = None
        self.scorer = Scorer(model_path)
        self.SCORE_TIME = 5    # 每隔多少秒进行一次打分
    
    def start_pomodoro(self, minutes):
        self.remaining_time = minutes * 60
        self.running = True
        self.timer_thread = threading.Thread(target=self._countdown)
        self.timer_thread.start()
    
    def _score(self):
        score = self.scorer.score(pic)
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
            print("时间到！")
            
    def start_forward(self):
        self.forward_time = 0
        self.running = True
        self.forward_thread = threading.Thread(target=self._forward_count)
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
    
    def stop(self):
        self.running = False
        print("计时已停止。")
        
        
if __name__ == "__main__":    
    model_path = "../model/emotion_net_best.pth"
    
    
    timer = PomodoroTimer(model_path)
    
    print("开始倒计时 (25分钟) ...")
    
    # timer.start_pomodoro(1)
    timer.start_forward()
    time.sleep(10)
    timer.stop()
    