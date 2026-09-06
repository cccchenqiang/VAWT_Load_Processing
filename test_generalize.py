"""测试通用化后的Web GUI API"""
import json
import urllib.request

BASE = "http://127.0.0.1:8080"

def post(path, data):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(path):
    with urllib.request.urlopen(f"{BASE}{path}") as resp:
        return json.loads(resp.read().decode("utf-8"))

# 1. 状态
print("=== 状态 ===")
print(get("/api/status"))

# 2. 启动extreme模式分析
print("\n=== 启动extreme模式分析 ===")
result = post("/api/analyze", {
    "filepath": r"D:\1worksfiles\py\laoshan\ultrapost\load_data_sample.txt",
    "mode": "extreme",
    "include_panel": False,
    "filter_enabled": True,
    "filter_cutoff": 5.0,
    "sn_m": 3.0,
    "sn_log_a": 12.0,
})
print(result)

# 3. 轮询
import time
for i in range(30):
    time.sleep(1)
    status = get("/api/status")
    if status["progress"] >= 100 or "错误" in status["status"]:
        print(f"  进度: {status['progress']}% - {status['status']}")
        break

# 4. 获取结果
print("\n=== 结果概览 ===")
results = get("/api/results")
print("文件信息:", results.get("file_info", {}).get("condition_type"), results.get("file_info", {}).get("condition_name"))
print("概览指标:", results.get("overview"))
if "extreme" in results:
    print("extreme结果存在, 叶片数:", len(results["extreme"].get("summary", [])))
if "eog" in results:
    print("eog兼容键存在, 叶片数:", len(results["eog"].get("summary", [])))
print("时序通道:", list(results.get("timeseries", {}).get("channels", {}).keys()))

print("\n通用化Web GUI API测试通过!")
