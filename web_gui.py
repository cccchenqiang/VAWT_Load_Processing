"""
垂直轴风轮载荷数据处理系统 - Web GUI
基于Python标准库http.server + ECharts前端，浏览器中操作

运行方式：
    python web_gui.py
    然后浏览器打开 http://localhost:8080
"""

import os
import math
import sys
import json
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Keep BLAS initialization bounded on Windows. Use VAWT_BLAS_THREADS only
# when a larger value has been tested safe for the target machine.
_blas_threads = os.environ.get("VAWT_BLAS_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = _blas_threads
os.environ["OMP_NUM_THREADS"] = _blas_threads
os.environ["MKL_NUM_THREADS"] = _blas_threads

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from bigfile_io import BigFileLoader, ConditionSlicer, quick_inspect
from preprocess import preprocess_dataframe
from extreme_load_analysis import run_extreme_analysis
from fatigue_analysis import run_fatigue_analysis
from blade_synthesis import synthesize_all_blades
from utils import export_to_excel, find_wind_speed_col
from validation import validate_load_data

# ============================================================
# 日志
# ============================================================
import logging
logger = logging.getLogger("vawt_web")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ============================================================
# 全局状态
# ============================================================
STATE = {
    "filepath": "",
    "results": None,
    "df": None,
    "mapper": None,
    "loader": None,
    "progress": 0,
    "status": "就绪",
    "is_running": False,
    # FAST 独立分析结果（不干扰 QBlade 流程）
    "fast_results": None,
    "fast_df": None,
    "fast_file": "",
}

PORT = 8080


# ============================================================
# 分析逻辑
# ============================================================
def run_analysis(filepath, mode="both", include_panel=False,
                 filter_enabled=True, filter_cutoff=5.0,
                 sn_m=3.0, sn_log_a=12.0):
    """运行完整分析，返回结果字典"""
    import time as _t
    _t0 = _t.time()
    # 更新配置
    config.preprocess.enable_filter = filter_enabled
    config.preprocess.filter_cutoff_hz = filter_cutoff
    config.fatigue.sn_m = sn_m
    config.fatigue.sn_log_a = sn_log_a

    STATE["progress"] = 5
    STATE["status"] = "解析文件..."

    loader = BigFileLoader(filepath)
    mapper = loader.mapper
    info = loader.info

    STATE["progress"] = 15
    STATE["status"] = "加载数据..."

    t_array = loader.get_time_array()
    sample_rate = info.sample_rate_hz
    df = loader.load_all(include_panel=include_panel)
    validation = validate_load_data(df, mapper)

    STATE["progress"] = 30
    STATE["status"] = "预处理..."

    df = preprocess_dataframe(df, config.preprocess, sample_rate)

    results = {
        "file_info": {
            "filename": os.path.basename(filepath),
            "size_mb": round(info.file_size_mb, 2),
            "condition_type": info.condition_type,
            "condition_name": info.condition_name,
            "num_rows": info.num_data_rows,
            "num_cols": info.num_columns,
            "num_blades": info.num_blades,
            "num_panels": info.num_panels,
            "sample_rate": round(sample_rate, 1),
            "duration": round(info.total_time, 2),
        },
        "overview": {},
        "validation": validation,
    }

    # 概览指标
    gc = config.global_ch
    if gc.inst_torque_col in df.columns:
        results["overview"]["torque_mean"] = round(float(df[gc.inst_torque_col].mean()), 1)
        results["overview"]["torque_max"] = round(float(df[gc.inst_torque_col].max()), 1)
    if gc.inst_thrust_col in df.columns:
        results["overview"]["thrust_mean"] = round(float(df[gc.inst_thrust_col].mean()), 1)
        results["overview"]["thrust_max"] = round(float(df[gc.inst_thrust_col].max()), 1)
    if gc.inst_power_col in df.columns:
        results["overview"]["power_mean_kw"] = round(float(df[gc.inst_power_col].mean() / 1000), 2)
        results["overview"]["power_max_kw"] = round(float(df[gc.inst_power_col].max() / 1000), 2)

    # 极限载荷分析（通用，支持EOG/NTM等所有风况）
    if mode in ("extreme", "eog", "both"):
        STATE["progress"] = 45
        STATE["status"] = f"极限载荷分析 ({info.condition_type})..."
        slicer = ConditionSlicer(loader)
        extreme_segments = slicer.detect_extreme_segments()
        extreme_results = run_extreme_analysis(
            df, mapper,
            extreme_segments=extreme_segments if extreme_segments else None,
            condition_type=info.condition_type,
        )
        results["extreme"] = _serialize_extreme(extreme_results)
        results["eog"] = results["extreme"]  # 向后兼容
        results["overview"]["critical_blade"] = extreme_results["wheel_results"].get("critical_blade", "-")

    # 疲劳分析
    if mode in ("fatigue", "both"):
        STATE["progress"] = 65
        STATE["status"] = "疲劳分析..."
        fatigue_results = run_fatigue_analysis(df, mapper, info.total_time)
        results["fatigue"] = _serialize_fatigue(fatigue_results)

    # 叶片合成
    STATE["progress"] = 80
    STATE["status"] = "叶片合成..."
    synth_results = synthesize_all_blades(df, mapper)
    results["synthesis"] = _serialize_synth(synth_results)

    # 塔顶载荷分析
    STATE["progress"] = 85
    STATE["status"] = "塔顶载荷分析..."
    try:
        from tower_top_analysis import run_tower_top_analysis
        tt = run_tower_top_analysis(df)
        results["tower_top"] = {
            "extreme": {k: {kk: (round(float(vv), 3) if isinstance(vv, (int, float, np.floating)) else vv)
                            for kk, vv in v.items()} for k, v in tt["extreme"].items()},
            "fatigue": {k: {kk: (round(float(vv), 3) if isinstance(vv, (int, float, np.floating)) else vv)
                            for kk, vv in v.items()} for k, v in tt["fatigue"].items()},
            "channels": tt.get("channels", {}),
            "time": [round(float(x), 3) for x in tt.get("time", [])],
        }
    except Exception as e:
        logger.warning(f"塔顶载荷分析失败: {e}")
        results["tower_top"] = {}

    # 叶片校核载荷（垂直轴：侧面固定，法向+切向+合成）
    if info.num_blades > 0:
        try:
            from load_envelope import analyze_case_for_envelope, blade_check_envelope
            env_input = analyze_case_for_envelope(df, mapper, info, case_id="single", weight=1.0)
            bc = blade_check_envelope([env_input], num_blades=info.num_blades,
                                      m=config.batch.fatigue_m.get("blade", 10))
            results["blade_check"] = _serialize_blade_check(bc)
        except Exception as e:
            logger.warning(f"叶片校核载荷分析失败: {e}")
            results["blade_check"] = {}

    # 时序数据（用于前端图表，降采样到500点）
    STATE["progress"] = 90
    STATE["status"] = "整理时序数据..."
    results["timeseries"] = _extract_timeseries(df, max_points=500)

    STATE["progress"] = 100
    STATE["status"] = "分析完成"

    # 扩展文件信息（创建时间、通道映射）
    results["file_info"]["creation_time"] = getattr(info, "creation_time", "--") or "--"
    results["file_info"]["calc_time"] = f"{_t.time() - _t0:.1f} s"
    # 通道映射信息
    results["mapper_info"] = {
        "num_global": len(getattr(mapper, "global_channels", []) or []),
        "num_blades": info.num_blades,
        "num_total_loads": sum(len(getattr(mapper, "blade_total_loads", {}).get(f"Blade_{n}", {}))
                               for n in range(1, info.num_blades + 1)),
        "num_panel": len(getattr(mapper, "blade_panel_loads", {}).get("Blade_1", {}).get("normal_force", [])) * info.num_blades if info.num_blades else 0,
    }

    STATE["results"] = results
    STATE["df"] = df
    STATE["mapper"] = mapper
    STATE["loader"] = loader
    STATE["calc_time"] = f"{_t.time() - _t0:.1f} s"

    return results


def _serialize_extreme(extreme_results):
    """序列化极限载荷结果为JSON友好格式"""
    out = {}
    if not extreme_results["summary_df"].empty:
        df = extreme_results["summary_df"]
        out["summary"] = df.to_dict(orient="records")
    else:
        out["summary"] = []  # 无叶片数据时保持空数组，避免前端.map报错
    # 整机载荷
    wheel = extreme_results["wheel_results"]
    out["wheel"] = {}
    for key in ["critical_blade", "critical_blade_load", "combined_peak_imbalance",
                "normal_peak_imbalance", "tangential_peak_imbalance"]:
        if key in wheel:
            val = wheel[key]
            out["wheel"][key] = float(val) if isinstance(val, (np.floating, float)) else val
    # 风轮推力/扭矩/功率/倾覆弯矩统计（含发生时刻）
    for key in ["thrust_stats", "torque_stats", "power_stats",
                "total_force_stats", "overturning_moment_stats"]:
        if key in wheel:
            out["wheel"][key] = {k: round(float(v), 2) for k, v in wheel[key].items()
                                 if isinstance(v, (int, float, np.floating))}
    # 最大/最小值发生时刻
    for key in ["thrust_max_time", "thrust_min_time", "torque_max_time", "torque_min_time",
                "power_max_time", "power_min_time", "thrust_max_value", "thrust_min_value",
                "torque_max_value", "torque_min_value", "power_max_value", "power_min_value"]:
        if key in wheel:
            val = wheel[key]
            out["wheel"][key] = round(float(val), 2) if isinstance(val, (int, float, np.floating)) else val
    return out


# 向后兼容别名
_serialize_eog = _serialize_extreme


def _serialize_fatigue(fatigue_results):
    """序列化疲劳结果"""
    out = {}
    if not fatigue_results["summary_df"].empty:
        df = fatigue_results["summary_df"]
        out["summary"] = df.to_dict(orient="records")
    else:
        out["summary"] = []  # 无叶片数据时保持空数组
    out["wheel"] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in fatigue_results["wheel_fatigue"].items()}
    # 载荷谱（每个叶片合成载荷）
    out["spectra"] = {}
    for bid, bres in fatigue_results.get("blade_results", {}).items():
        if "combined_load" in bres:
            ch = bres["combined_load"]
            if "spectrum_ranges" in ch and len(ch["spectrum_ranges"]) > 0:
                out["spectra"][bid] = {
                    "ranges": [round(float(x), 2) for x in ch["spectrum_ranges"]],
                    "counts": [float(x) for x in ch["spectrum_counts"]],
                }
    return out


def _serialize_synth(synth_results):
    """序列化合成结果"""
    out = {}
    for key in ["imbalance_normal", "imbalance_tangential"]:
        if key in synth_results:
            imb = synth_results[key]
            out[key] = {
                "imbalance_mean": round(float(imb.get("imbalance_mean", 0)), 4),
                "imbalance_max": round(float(imb.get("imbalance_max", 0)), 4),
            }
    if "pulsation" in synth_results:
        p = synth_results["pulsation"]
        out["pulsation"] = {
            "avg_rpm": round(float(p.get("avg_rpm", 0)), 2),
            "pulsation_ratio": round(float(p.get("pulsation_ratio", 0)), 4),
        }
    return out


def _serialize_blade_check(blade_check):
    """序列化叶片校核载荷（极限包络+疲劳包络）"""
    out = {}
    for bid, benv in blade_check.items():
        ext = benv.get("extreme", [])
        fat = benv.get("fatigue", [])
        out[bid] = {
            "extreme": _df_to_records(ext),
            "fatigue": _df_to_records(fat),
        }
    return out


def _df_to_records(df):
    """DataFrame转records，NaN转None，兼容JSON"""
    if df is None or df.empty:
        return []
    try:
        return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
    except Exception:
        return df.to_dict(orient="records")


def _extract_timeseries(df, max_points=500):
    """提取关键通道时序数据（降采样）"""
    time_col = config.global_ch.time_col
    if time_col not in df.columns:
        return {}

    t = df[time_col].values
    n = len(t)
    step = max(1, n // max_points)
    t_sampled = t[::step].tolist()

    channels = {}
    # 风速列：兼容 Inflow/Meas 列名差异（B3修复）
    wind_col = find_wind_speed_col(df.columns)
    key_channels = [
        ("Momentary_Aerodynamic_Torque_[Nm]", "torque"),
        ("Momentary_Aerodynamic_Thrust_[N]", "thrust"),
        ("Momentary_Aerodynamic_Power_[W]", "power"),
        (wind_col, "wind_speed"),
        ("Rotational_Speed_[rpm]", "rpm"),
        ("Total_Normal_Load_Blade_1_[N]", "blade1_normal"),
        ("Total_Normal_Load_Blade_2_[N]", "blade2_normal"),
        ("Total_Normal_Load_Blade_3_[N]", "blade3_normal"),
        ("Total_Tangential_Load_Blade_1_[N]", "blade1_tangential"),
    ]
    for col, name in key_channels:
        if col in df.columns:
            vals = df[col].values[::step]
            channels[name] = [round(float(x), 3) for x in vals]

    return {"time": [round(float(x), 3) for x in t_sampled], "channels": channels}


# ============================================================
# HTTP请求处理
# ============================================================
class RequestHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def log_message(self, format, *args):
        pass  # 静默日志

    def _send_json(self, data, status=200):
        # 递归清理NaN/Infinity，转换为null（JSON不支持NaN）
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            if isinstance(obj, (np.floating,)):
                v = float(obj)
                if np.isnan(v) or np.isinf(v):
                    return None
                return v
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return clean(obj.tolist())
            return obj
        body = json.dumps(clean(data), ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html(FRONTEND_HTML)
        elif path == "/api/status":
            self._send_json({
                "progress": STATE["progress"],
                "status": STATE["status"],
                "is_running": STATE["is_running"],
                "has_results": STATE["results"] is not None,
            })
        elif path == "/api/results":
            if STATE["results"]:
                self._send_json(STATE["results"])
            else:
                self._send_json({"error": "暂无结果"}, 404)
        elif path == "/api/channels":
            if STATE["df"] is not None:
                cols = [c for c in STATE["df"].columns if not c.startswith("PAN_")]
                self._send_json({"channels": cols})
            else:
                self._send_json({"channels": []})
        elif path == "/api/fast_channels":
            self._handle_fast_channels()
        elif path == "/api/fast_results":
            self._handle_fast_results()
        elif path == "/api/list_dir":
            self._handle_list_dir(parsed)
        elif path == "/api/drives":
            self._handle_drives()
        elif path == "/api/download":
            self._handle_download(parsed)
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_download(self, parsed):
        """文件下载"""
        from urllib.parse import parse_qs, unquote
        query = parse_qs(parsed.query)
        fp = query.get("path", [""])[0]
        fp = unquote(fp)
        if not fp or not os.path.exists(fp):
            self._send_json({"error": "文件不存在"}, 404)
            return
        try:
            with open(fp, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            # 中文文件名：RFC 5987 编码，避免 latin-1 头编码异常导致协议冲突
            from urllib.parse import quote
            fname = os.path.basename(fp)
            try:
                fname.encode("ascii")
                cd = 'attachment; filename="' + fname + '"'
            except UnicodeEncodeError:
                ascii_name = fname.encode("ascii", "ignore").decode("ascii") or "download"
                cd = ('attachment; filename="' + ascii_name + '"; filename*=UTF-8\'\'' + quote(fname))
            self.send_header("Content-Disposition", cd)
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            data = {}

        if path == "/api/preview":
            self._handle_preview(data)
        elif path == "/api/analyze":
            self._handle_analyze(data)
        elif path == "/api/fast_analyze":
            self._handle_fast_analyze(data)
        elif path == "/api/fast_channels":
            self._handle_fast_channels()
        elif path == "/api/fast_channel_data":
            self._handle_fast_channel_data(data)
        elif path == "/api/export":
            self._handle_export(data)
        elif path == "/api/channel_data":
            self._handle_channel_data(data)
        elif path == "/api/export_channels":
            self._handle_export_channels(data)
        elif path == "/api/export_report":
            self._handle_export_report(data)
        elif path == "/api/batch":
            self._handle_batch(data)
        elif path == "/api/batch_report":
            self._handle_batch_report(data)
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_list_dir(self, parsed):
        """列出指定目录的内容（用于文件浏览器）"""
        from urllib.parse import parse_qs
        query = parse_qs(parsed.query)
        path = query.get("path", [""])[0]

        # 默认路径：用户主目录
        if not path or not os.path.exists(path):
            path = os.path.expanduser("~")

        try:
            entries = []
            # 上级目录
            parent = os.path.dirname(path.rstrip(os.sep))
            if parent and parent != path:
                entries.append({"name": "..", "path": parent, "is_dir": True})

            # 当前目录内容
            items = sorted(os.listdir(path), key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
            for name in items:
                full = os.path.join(path, name)
                try:
                    is_dir = os.path.isdir(full)
                    size = os.path.getsize(full) if not is_dir else 0
                    entries.append({
                        "name": name,
                        "path": full,
                        "is_dir": is_dir,
                        "size": size,
                    })
                except (PermissionError, OSError):
                    continue

            self._send_json({
                "current": path,
                "parent": parent,
                "entries": entries,
            })
        except Exception as e:
            self._send_json({"error": str(e), "current": path}, 500)

    def _handle_drives(self):
        """获取Windows盘符列表"""
        import string
        drives = []
        if os.name == "nt":
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
        else:
            drives = ["/"]
        self._send_json({"drives": drives})

    def _handle_preview(self, data):
        filepath = data.get("filepath", "")
        if not filepath or not os.path.exists(filepath):
            self._send_json({"error": "文件不存在"}, 400)
            return
        try:
            info = quick_inspect(filepath)
            self._send_json({"info": info})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_analyze(self, data):
        if STATE["is_running"]:
            self._send_json({"error": "分析正在进行中"}, 409)
            return

        filepath = data.get("filepath", "")
        if not filepath or not os.path.exists(filepath):
            self._send_json({"error": "文件不存在"}, 400)
            return

        def worker():
            try:
                STATE["is_running"] = True
                run_analysis(
                    filepath=filepath,
                    mode=data.get("mode", "both"),
                    include_panel=data.get("include_panel", False),
                    filter_enabled=data.get("filter_enabled", True),
                    filter_cutoff=data.get("filter_cutoff", 5.0),
                    sn_m=data.get("sn_m", 3.0),
                    sn_log_a=data.get("sn_log_a", 12.0),
                )
            except Exception as e:
                STATE["status"] = f"错误: {e}"
                STATE["progress"] = 0
                traceback.print_exc()
            finally:
                STATE["is_running"] = False

        threading.Thread(target=worker, daemon=True).start()
        self._send_json({"started": True})

    # ---- FAST .out 独立分析（不影响 QBlade）----
    def _handle_fast_analyze(self, data):
        """FAST .out 文件独立分析"""
        if STATE["is_running"]:
            self._send_json({"error": "分析正在进行中"}, 409)
            return
        filepath = data.get("filepath", "")
        if not filepath or not os.path.exists(filepath):
            self._send_json({"error": "文件不存在"}, 400)
            return

        def worker():
            try:
                STATE["is_running"] = True
                STATE["progress"] = 5
                STATE["status"] = "解析 FAST 文件..."
                from fast_io import parse_fast_header, load_fast_data
                from fast_analysis import run_fast_analysis
                info = parse_fast_header(filepath)
                STATE["progress"] = 20
                STATE["status"] = "加载 FAST 数据..."
                df = load_fast_data(filepath, header=info)
                STATE["progress"] = 40
                STATE["status"] = "FAST 载荷分析..."
                res = run_fast_analysis(filepath, info=info)
                STATE["fast_results"] = res
                STATE["fast_df"] = df
                STATE["fast_file"] = filepath
                STATE["progress"] = 100
                STATE["status"] = "分析完成"
            except Exception as e:
                STATE["status"] = f"错误: {e}"
                STATE["progress"] = 0
                traceback.print_exc()
            finally:
                STATE["is_running"] = False

        threading.Thread(target=worker, daemon=True).start()
        self._send_json({"started": True})

    def _handle_fast_results(self):
        """返回 FAST 分析结果"""
        if STATE["fast_results"] is None:
            self._send_json({"error": "尚未进行 FAST 分析"})
            return
        self._send_json(STATE["fast_results"])

    def _handle_fast_channels(self):
        """返回 FAST 数据全部列名"""
        if STATE["fast_df"] is None:
            self._send_json({"channels": []})
            return
        self._send_json({"channels": list(STATE["fast_df"].columns)})

    def _handle_fast_channel_data(self, data):
        """返回 FAST 通道时序数据（降采样+统计）

        channel 支持特殊键(thrust/torque/power/rpm/tsr/wind_speed)或原始列名。
        """
        channel = data.get("channel", "")
        if STATE["fast_df"] is None:
            self._send_json({"error": "尚未加载 FAST 数据"}, 400)
            return
        from fast_analysis import fast_channel_vector
        from fast_io import FASTChannelMapper
        df = STATE["fast_df"]
        if "Time" not in df.columns:
            self._send_json({"error": "FAST 数据缺少 Time 列"}, 400)
            return
        mapper = STATE.get("fast_mapper")
        if mapper is None:
            mapper = FASTChannelMapper(list(df.columns),
                                       STATE["fast_results"].get("file_info", {}).get("num_blades", 0) if STATE.get("fast_results") else 0,
                                       STATE["fast_results"].get("file_info", {}).get("num_nodes", 0) if STATE.get("fast_results") else 0)
            STATE["fast_mapper"] = mapper
        vals, label = fast_channel_vector(df, mapper, channel)
        if vals is None:
            self._send_json({"error": f"通道不可用: {channel}"}, 400)
            return
        vals = np.asarray(vals, dtype=float)
        t = df["Time"].to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            self._send_json({"error": "通道数据全为无效值"}, 400)
            return
        step = max(1, len(t) // 500)
        max_idx_full = int(np.nanargmax(vals))
        min_idx_full = int(np.nanargmin(vals))
        idx = list(range(0, len(t), step))
        if max_idx_full not in idx:
            idx.append(max_idx_full)
        if min_idx_full not in idx:
            idx.append(min_idx_full)
        idx = sorted(idx)
        self._send_json({
            "time": [round(float(x), 3) for x in t[idx]],
            "values": [None if math.isnan(float(x)) else round(float(x), 3) for x in vals[idx]],
            "mean": round(float(np.nanmean(vals)), 2),
            "max": round(float(np.nanmax(vals)), 2),
            "min": round(float(np.nanmin(vals)), 2),
            "std": round(float(np.nanstd(vals)), 2),
            "max_idx": idx.index(max_idx_full),
            "min_idx": idx.index(min_idx_full),
            "label": label,
        })

    def _handle_export(self, data):
        if not STATE["results"]:
            self._send_json({"error": "暂无结果可导出"}, 400)
            return
        try:
            export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(export_dir, exist_ok=True)
            filename = f"载荷分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = os.path.join(export_dir, filename)

            sheets = {}
            res = STATE["results"]
            ext_data = res.get("extreme") or res.get("eog")
            if ext_data and "summary" in ext_data:
                sheets["极限载荷叶片极值"] = pd.DataFrame(ext_data["summary"])
            if "fatigue" in res and "summary" in res["fatigue"]:
                sheets["疲劳损伤汇总"] = pd.DataFrame(res["fatigue"]["summary"])
            if sheets:
                export_to_excel(sheets, filepath)
                self._send_json({"filepath": filepath, "filename": filename})
            else:
                self._send_json({"error": "没有可导出的数据"}, 400)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_channel_data(self, data):
        """获取指定通道的时序数据"""
        channel = data.get("channel", "")
        if STATE["df"] is None or channel not in STATE["df"].columns:
            self._send_json({"error": "通道不存在"}, 400)
            return
        time_col = config.global_ch.time_col
        t = STATE["df"][time_col].values
        vals = STATE["df"][channel].values
        # 降采样（仅用于绘制曲线，统计仍基于全量原始数据）
        step = max(1, len(t) // 500)
        # 全量数据真实最值索引
        max_idx_full = int(np.argmax(vals))
        min_idx_full = int(np.argmin(vals))
        # 采样索引集合，并确保全局最值点被包含在曲线中，
        # 使图上最大值/最小值标注能精确落在曲线峰值/谷值上
        idx = list(range(0, len(t), step))
        if max_idx_full not in idx:
            idx.append(max_idx_full)
        if min_idx_full not in idx:
            idx.append(min_idx_full)
        idx = sorted(idx)
        t_sampled = t[idx]
        v_sampled = vals[idx]
        self._send_json({
            "time": [round(float(x), 3) for x in t_sampled],
            "values": [round(float(x), 3) for x in v_sampled],
            "mean": round(float(np.mean(vals)), 2),
            "max": round(float(np.max(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "std": round(float(np.std(vals)), 2),
            "max_idx": idx.index(max_idx_full),
            "min_idx": idx.index(min_idx_full),
        })

    def _handle_export_channels(self, data):
        """载荷数据导出：多通道 → Excel/CSV/TXT"""
        try:
            from report_export import export_channels

            channels = data.get("channels", []) or []
            fmt = data.get("format", "excel")
            filename = data.get("filename", "")
            out_dir = data.get("dir", "")
            if STATE["df"] is None:
                self._send_json({"error": "请先运行分析加载数据"}, 400)
                return
            if not channels:
                self._send_json({"error": "请至少选择一个通道"}, 400)
                return
            # 使用项目名称作为默认前缀
            project = data.get("project_name", "")
            prefix = project or "载荷数据"
            filepath, filename = export_channels(
                STATE["df"], channels, fmt,
                output_dir=out_dir or None, prefix=prefix,
                filename=filename or None)
            self._send_json({"filepath": filepath, "filename": filename})
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _handle_export_report(self, data):
        """分析结果报告导出：PDF/Word"""
        try:
            from report_export import generate_report

            fmt = data.get("format", "pdf")
            filename = data.get("filename", "")
            out_dir = data.get("dir", "")
            if not STATE["results"]:
                self._send_json({"error": "请先完成分析"}, 400)
                return
            filepath, filename = generate_report(
                STATE["results"], STATE["df"], fmt,
                output_dir=out_dir or None,
                calc_time=STATE.get("calc_time", "--"),
                filename=filename or None)
            self._send_json({"filepath": filepath, "filename": filename})
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)


# ============================================================
# 前端HTML
# ============================================================

    def _handle_batch(self, data):
        """批处理分析（多工况DLC）"""
        import time as _t
        from batch_processing import run_batch, serialize_batch_result

        directory = data.get("directory", "")
        cases_file = data.get("cases_file", "")
        include_panel = bool(data.get("include_panel", False))
        # B1修复：从界面读取叶片/塔顶疲劳指数 m（前端输入框真实生效）
        blade_m = float(data.get("blade_m", config.batch.fatigue_m.get("blade", 10)))
        tower_m = float(data.get("tower_m", config.batch.fatigue_m.get("tower", 3)))

        # 收集工况列表（目录扫描 或 cases.yaml）
        from batch_processing import CaseScanner
        scanner = CaseScanner()
        cases = scanner.resolve_cases(directory=directory, cases_file=cases_file)
        if not cases:
            self._send_json({"error": "未找到工况文件，请选择目录或提供cases.yaml"}, 400)
            return

        # 过滤仅保留存在的文件
        cases = [c for c in cases if os.path.exists(c.get("file", ""))]
        if not cases:
            self._send_json({"error": "所选目录下没有数据文件(.txt/.dat)"}, 400)
            return

        STATE["is_running"] = True
        STATE["status"] = "批处理分析中..."
        STATE["progress"] = 5

        try:
            def _cb(done, total, msg):
                STATE["progress"] = int(5 + 90 * done / total)
                STATE["status"] = f"批处理 {done}/{total}: {msg}"

            _t0 = _t.time()
            batch_result = run_batch(cases,
                                     include_panel=include_panel,
                                     progress_cb=_cb,
                                     blade_m=blade_m, tower_m=tower_m)
            ser = serialize_batch_result(batch_result)
            ser["calc_time"] = f"{_t.time() - _t0:.1f} s"
            STATE["batch_results"] = ser
            STATE["progress"] = 100
            STATE["status"] = "批处理完成"
            self._send_json(ser)
        except Exception as e:
            STATE["status"] = f"批处理错误: {e}"
            self._send_json({"error": str(e)}, 500)
        finally:
            STATE["is_running"] = False

    def _handle_batch_report(self, data):
        """导出批处理报告（PDF/Word）"""
        from report_export import generate_batch_report

        fmt = data.get("format", "pdf")
        filename = data.get("filename", "")
        output_dir = data.get("output_dir", "")

        batch_result = STATE.get("batch_results")
        if not batch_result:
            self._send_json({"error": "请先运行批处理分析"}, 400)
            return

        try:
            fp, fn = generate_batch_report(batch_result, fmt=fmt,
                                           output_dir=output_dir or None,
                                           calc_time=batch_result.get("calc_time"),
                                           filename=filename or None)
            logger.info(f"批处理报告已导出: {fn}")
            self._send_json({"ok": True, "filepath": fp, "filename": fn})
        except Exception as e:
            self._send_json({"error": f"报告导出失败: {e}"}, 500)

def _load_frontend_html():
    """从独立 frontend.html 文件加载前端页面（O3：前端与后端分离，便于维护）"""
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend.html")
    try:
        with open(_p, "r", encoding="utf-8") as _f:
            return _f.read()
    except FileNotFoundError:
        return "<html><body><h3>缺少 frontend.html 文件（请与 web_gui.py 放在同一目录）</h3></body></html>"


FRONTEND_HTML = _load_frontend_html()


def main():
    server = HTTPServer(("127.0.0.1", PORT), RequestHandler)
    url = f"http://127.0.0.1:{PORT}"
    print("=" * 60)
    print("垂直轴风轮载荷数据处理系统 - Web GUI")
    print("=" * 60)
    print(f"服务器已启动: {url}")
    print(f"按 Ctrl+C 停止服务器")
    print("=" * 60)

    # 自动打开浏览器
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.server_close()


if __name__ == "__main__":
    main()
