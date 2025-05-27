import vosk
import speech_recognition as sr
import re

class SpeechRecognizer:
    def __init__(self):
        self.r = sr.Recognizer()
        
    def _parse_chinese_number(self, cn: str) -> int:
        """ 中文数字转整数 """
        numeral_map = {
            '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
        }
        unit_map = {
            '十': 10, '百': 100, '千': 1000
        }

        result = 0
        unit = 1
        num = 0
        i = len(cn) - 1

        while i >= 0:
            char = cn[i]
            if char in numeral_map:
                num = numeral_map[char]
                result += num * unit
                i -= 1
            elif char in unit_map:
                unit = unit_map[char]
                # 如果前一位是单位（例如“十”前没有“一”时默认为1）
                if i == 0 or cn[i - 1] not in numeral_map:
                    result += unit
                i -= 1
            else:
                i -= 1

        return result if result != 0 else 1
    
    def process_result(self, text):
        """处理识别结果，返回对应的模式和时长"""
        # 删除所有非中文字符
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

        # 处理“开始专注”
        if re.search(r"开始专注", text):
            return ("forward", 0)

        # 处理“专注X分钟”
        match = re.search(r'专注(.{1,3})分钟', text)
        if match:
            num_text = match.group(1)
            number = self._parse_chinese_number(num_text)
            return ("pomodoro", number)

        # 默认返回
        return ("pomodoro", 1)
    
    def record(self, duration=5):
        with sr.Microphone() as source:
            print("开始录音")
            audioData = self.r.listen(source, duration)
        print("录音结束")
        said = self.r.recognize_vosk(audioData, language='zh-CN')
        print(said)
        return self.process_result(said)
        
if __name__ == "__main__":
    speech = SpeechRecognizer()
    # print(speech._parse_chinese_number("三十六"))
    mode, time = speech.record()
    print(f"专注模式：{mode}, 时间：{time}")