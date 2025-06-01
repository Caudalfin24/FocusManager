from flask import Flask, render_template
import data
app = Flask(__name__)

records = data.get_records()
light, noise = data.get_sensor()

@app.route("/")
def index():
    return render_template('index.html',
                           records=records,
                           light_value=light,       # 模拟光照值（lx）
                           noise_value=noise)      # 模拟噪声值（dB）

@app.route("/detail/<int:record_id>")
def detail(record_id):
    record = next((r for r in records if r["id"] == record_id), None)
    if record is None:
        return "记录未找到", 404
    return render_template("detail.html", record=record)

if __name__ == "__main__":
    app.run(debug=False)