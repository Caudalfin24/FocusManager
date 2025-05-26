from flask import Flask, render_template
from data import records

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html',
                           records=records,
                           light_value=520,       # 模拟光照值（lx）
                           noise_value=38.7)      # 模拟噪声值（dB）

@app.route("/detail/<int:record_id>")
def detail(record_id):
    record = next((r for r in records if r["id"] == record_id), None)
    if record is None:
        return "记录未找到", 404
    return render_template("detail.html", record=record)

if __name__ == "__main__":
    app.run(debug=True)