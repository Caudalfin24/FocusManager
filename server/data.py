import requests
from datetime import datetime

pid = "8I1YkXJG8I"
token = "version=2018-10-31&res=products%2F8I1YkXJG8I%2Fdevices%2Fraspberrypi&et=1830307200&method=sha1&sign=4c3f1XUi%2BJHEv%2FVQZc2siJN4G6Q%3D"
device_id = "raspberrypi"

def get_records():
    url = "https://iot-api.heclouds.com/datapoint/history-datapoints"
    headers = {
        "Authorization": token
    }
    params = {
        "product_id": pid,
        "device_name": device_id,
        "limit": 100
    }
    response = requests.get(url, headers=headers, params=params)
    response_datas = response.json()['data']['datastreams'][2]['datapoints']
    
    records = []
    cnt = 1
    for data in response_datas:
        value = data['value']
        if not isinstance(value.get('scores'), list):
            print("'scores' 不是数组" )
            continue
        
        time_str = value.get('time')
        try:
            time_obj = datetime.fromisoformat(time_str)
            dt = time_obj.strftime("%Y-%m-%d %H:%M:%S")
            print("转换成功")
        
            record = {
                "id": cnt,
                "type": value.get('type'),
                "duration": value.get('length'),
                "datetime": dt,
                "focus_scores": value.get('scores')
            }
            records.append(record)
            cnt += 1
        except TypeError or ValueError:
            print("'time' 不是有效的 ISO 时间戳格式")
            continue
    print(records)
    return records
