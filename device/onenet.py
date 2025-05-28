import paho.mqtt.client as mqtt
import json

""" 参数设置 """

pid = "8I1YkXJG8I"
token = "version=2018-10-31&res=products%2F8I1YkXJG8I%2Fdevices%2Fraspberrypi&et=1830307200&method=sha1&sign=4c3f1XUi%2BJHEv%2FVQZc2siJN4G6Q%3D"
device_id = "raspberrypi"
broker = "mqtts.heclouds.com"

publish_topic = f"$sys/{pid}/{device_id}/dp/post/json"

# 接收响应的主题（可选）
accepted_topic = f"$sys/{pid}/{device_id}/dp/post/json/accepted"
rejected_topic = f"$sys/{pid}/{device_id}/dp/post/json/rejected"

# 回调函数：连接成功时触发
def on_connect(client, userdata, flags, rc):
    print("已连接，返回码："+str(rc))
    # 订阅主题
    if rc == 0:
        print("连接成功，开始上报数据点...")
        # 订阅响应主题
        client.subscribe(accepted_topic)
        client.subscribe(rejected_topic)
    else:
        print("连接失败")

# 回调函数：收到消息时触发
def on_message(client, userdata, msg):
    print(f"收到消息：主题={msg.topic}，内容={msg.payload.decode()}")


class OneNet:
    def __init__(self):
        self.client = mqtt.Client(client_id=device_id)  # Client ID 必须唯一
        # 设置用户名和密码
        self.client.username_pw_set(username=pid, password=token)

        # 绑定回调函数
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        


    def report_data(self, scores, time, length, record_type):
        """上报专注数据"""
        # 连接 MQTT 服务器
        self.client.connect(broker, 1883, 60) 
        # 循环等待消息
        self.client.loop_start()
        # 构造数据
        datapack = {
            "scores": scores,
            "time": time,
            "length": length,
            "type": record_type
        }
        data = json.dumps(datapack)
        print(data)
        payload = {
            "id": 123,
            "dp": {
                "records": [{
                    "v": datapack
                }],
            }
        }
        # 发布数据
        self.client.publish(publish_topic, json.dumps(payload), qos=1)
        print("已上报数据:", payload)
        self.client.disconnect()
    
    def report_light(self, value):
        # 连接 MQTT 服务器
        self.client.connect(broker, 1883, 60) 
        # 循环等待消息
        self.client.loop_start()
        payload = {
            "id": 123,
            "dp": {
                "light": [{
                    "v": value
                }],
            }
        }
        # 发布数据
        self.client.publish(publish_topic, json.dumps(payload), qos=1)
        print("已上报数据:", payload)
        self.client.disconnect()
        
    def report_noise(self, value):
        # 连接 MQTT 服务器
        self.client.connect(broker, 1883, 60) 
        # 循环等待消息
        self.client.loop_start()
        payload = {
            "id": 123,
            "dp": {
                "noise": [{
                    "v": value
                }],
            }
        }
        # 发布数据
        self.client.publish(publish_topic, json.dumps(payload), qos=1)
        print("已上报数据:", payload)
        self.client.disconnect()

# if __name__ =="__main__":
#     from datetime import datetime
    
#     onenet = OneNet()
#     record_scores = [23.5, 25.1, 22.8]       
#     record_time = datetime.now().isoformat() 
#     record_length = 25
#     onenet.report_data(record_scores, record_time, record_length)