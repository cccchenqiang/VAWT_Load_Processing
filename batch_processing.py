"""
多工况批处理调度器（IEC 61400 DLC）

功能：
1. 目录扫描/配置清单识别工况
2. 逐文件运行分析流水线（复用现有模块）
3. 并发控制（大文件内存受限，默认2-3）
4. 汇总生成载荷包络
5. 进度/日志管理

数据流：
多个工况文件 → 批量识别 → 逐文件分析 → 载荷包络 → 部件校核输入
"""

import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import config
from bigfile_io import BigFileLoader
from preprocess import preprocess_dataframe
from load_envelope import analyze_case_for_envelope, build_envelope

logger = logging.getLogger(__name__)


# ============================================================
# 工况识别
# ============================================================
class CaseScanner:
    """工况识别器：自动扫描目录 / 解析cases.yaml"""

    def __init__(self, cfg=None):
        self.cfg = cfg or config.batch

    def scan_directory(self, directory: str) -> List[Dict]:
        """扫描目录下的所有载荷文件（.txt/.dat）"""
        cases = []
        if not os.path.isdir(directory):
            return cases
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith((".txt", ".dat")):
                fp = os.path.join(directory, fname)
                cases.append(self._with_metadata({"file": fp}, fname))
        return cases

    def load_cases_file(self, yaml_path: str) -> List[Dict]:
        """加载工况清单，支持三种格式（按扩展名自动识别）：
        - .yaml / .yml : 结构化配置（需安装 pyyaml）
        - .csv         : file,weight,extreme 表头
        - .txt         : 每行 文件路径,权重,extreme(可选)
        返回: [{"file":..., "weight":..., "extreme":...}]
        """
        cases = []
        if not os.path.exists(yaml_path):
            return cases
        ext = os.path.splitext(yaml_path)[1].lower()
        try:
            if ext in (".yaml", ".yml"):
                import yaml
                with open(yaml_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                cases = data.get("cases", [])
            elif ext == ".csv":
                import csv
                with open(yaml_path, encoding="utf-8-sig", newline="") as f:
                    rows = [r for r in csv.reader(f)
                            if r and r[0].strip() and not r[0].strip().startswith("#")]
                if not rows:
                    return cases
                header = [h.strip().lower() for h in rows[0]]
                for row in rows[1:]:
                    if not row or not row[0].strip():
                        continue
                    rec = dict(zip(header, [x.strip() for x in row] + [""] * (len(header) - len(row))))
                    if not rec.get("file"):
                        continue
                    cases.append({
                        "file": rec["file"],
                        "weight": float(rec["weight"]) if rec.get("weight") else 1.0,
                        "extreme": rec.get("extreme", "").lower() in ("1", "true", "yes", "y"),
                    })
            elif ext == ".txt":
                with open(yaml_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # 支持逗号 / 分号 / Tab / 空格 分隔
                        parts = [p.strip() for p in line.replace(";", ",").replace("\t", ",").split(",")]
                        parts = [p for p in parts if p]
                        if not parts:
                            continue
                        case = {"file": parts[0]}
                        if len(parts) > 1:
                            try:
                                case["weight"] = float(parts[1])
                            except ValueError:
                                pass
                        if len(parts) > 2:
                            case["extreme"] = parts[2].lower() in ("1", "true", "yes", "y", "extreme")
                        cases.append(case)
            else:
                logger.error(f"不支持的工况清单格式: {ext}（支持 .yaml/.csv/.txt）")
                return cases
            # 相对路径转绝对 + 默认值
            base = os.path.dirname(os.path.abspath(yaml_path))
            for c in cases:
                if "file" in c and not os.path.isabs(c["file"]):
                    c["file"] = os.path.join(base, c["file"])
                c["_weight"] = c.get("weight", 1.0)
                c["_extreme"] = c.get("extreme", False)
                self._with_metadata(c, os.path.basename(c["file"]))
        except ImportError:
            logger.error("加载 .yaml 需安装 pyyaml；可改用 cases.csv / cases.txt 格式（无需额外依赖）")
        except Exception as e:
            logger.error(f"加载工况清单失败: {e}")
        return cases

    def _with_metadata(self, case: Dict, filename: str) -> Dict:
        """补齐工况类型、DLC编号和显示名称，支持文件名自动识别。"""
        raw_type = str(case.get("condition_type") or case.get("type") or "").strip()
        if not raw_type:
            match = re.search(r"(?i)(EOG|EWM|NTM|NWP|EWS|ECD|DLC\d*|STEADY|TURB)", filename)
            raw_type = match.group(1).upper() if match else "UNKNOWN"
        raw_type = raw_type.upper().replace("-", "_")
        dlc = str(case.get("dlc") or self.cfg.dlc_map.get(raw_type, "")).strip()
        case["condition_type"] = raw_type
        case["dlc"] = dlc or "未分类"
        case["label"] = case.get("label") or os.path.splitext(filename)[0]
        case.setdefault("weight", 1.0)
        case.setdefault("extreme", False)
        return case

    def resolve_cases(self, directory: str = "", cases_file: str = "") -> List[Dict]:
        """解析工况列表（配置清单优先，否则扫描目录）"""
        cases = []
        if cases_file and os.path.exists(cases_file):
            cases = self.load_cases_file(cases_file)
        elif directory:
            cases = self.scan_directory(directory)
        return cases


# ============================================================
# 单工况分析
# ============================================================
def analyze_single_case(filepath: str, include_panel: bool = False,
                        condition_type_hint: str = "",
                        blade_m: Optional[float] = None,
                        tower_m: Optional[float] = None) -> Dict:
    """
    分析单个工况文件，提取包络所需数据
    返回: analyze_case_for_envelope() 的结果 + file_info
    """
    loader = BigFileLoader(filepath)
    mapper = loader.mapper
    info = loader.info

    # 加载时间序列以更新采样率/总时长（parse_qblade_header阶段为0）
    loader.get_time_array()

    df = loader.load_all(include_panel=include_panel)
    df = preprocess_dataframe(df, config.preprocess, info.sample_rate_hz)

    case_id = os.path.splitext(os.path.basename(filepath))[0]
    cond_type = info.condition_type or condition_type_hint

    result = analyze_case_for_envelope(df, mapper, info, case_id=case_id,
                                       weight=1.0, condition_type=cond_type,
                                       blade_m=blade_m, tower_m=tower_m)
    result["file_info"] = {
        "filename": os.path.basename(filepath),
        "size_mb": round(info.file_size_mb, 2),
        "condition_type": cond_type,
        "condition_name": info.condition_name,
        "num_rows": info.num_data_rows,
        "num_cols": info.num_columns,
        "num_blades": info.num_blades,
        "sample_rate": round(info.sample_rate_hz, 1),
    }
    result["_weight"] = 1.0
    return result


# ============================================================
# 批处理主入口
# ============================================================
def run_batch(cases: List[Dict], include_panel: bool = False,
              progress_cb=None,
              blade_m: Optional[float] = None,
              tower_m: Optional[float] = None) -> Dict:
    """
    运行批处理（B1修复：去掉无效的 n_workers 并发参数，改为串行以避免大文件内存溢出；
    blade_m/tower_m 从界面传入，控制叶片/塔顶疲劳指数 m）
    cases: [{"file": path, "weight": w, ...}]
    返回: {cases: [各工况结果], envelope: 包络结果}
    """
    num_blades = 0
    case_results = []
    failed_cases = []
    total = len(cases)

    # 串行模式（大文件内存考虑：每份几百MB~2GB，避免并发内存溢出）
    for i, case in enumerate(cases):
        fp = case["file"]
        try:
            if progress_cb:
                progress_cb(i + 1, total, f"分析 {os.path.basename(fp)}")
            res = analyze_single_case(
                fp,
                include_panel=include_panel,
                condition_type_hint=case.get("condition_type", ""),
                blade_m=blade_m,
                tower_m=tower_m,
            )
            # 应用权重
            res["weight"] = float(case.get("weight", case.get("_weight", 1.0)))
            res["_extreme"] = case.get("extreme", case.get("_extreme", False))
            res["dlc"] = case.get("dlc", config.batch.dlc_map.get(
                str(res.get("file_info", {}).get("condition_type", "")).upper(), "未分类"))
            res["label"] = case.get("label") or res.get("case_id")
            case_results.append(res)
            num_blades = max(num_blades, int(res["file_info"].get("num_blades", 0)))
        except Exception as e:
            logger.error(f"工况 {fp} 分析失败: {e}")
            failed_cases.append({
                "file": fp,
                "filename": os.path.basename(fp),
                "error": str(e),
                "condition_type": case.get("condition_type", "UNKNOWN"),
                "dlc": case.get("dlc", "未分类"),
            })
            if progress_cb:
                progress_cb(i + 1, total, f"失败 {os.path.basename(fp)}: {e}")
            continue

    if not case_results:
        return {"error": "所有工况分析失败", "cases": [], "envelope": None}

    # 构建包络（num_blades=0 表示无叶片数据，不生成叶片包络）
    if blade_m is None:
        blade_m = config.batch.fatigue_m.get("blade", 10)
    if tower_m is None:
        tower_m = config.batch.fatigue_m.get("tower", 3)
    envelope = build_envelope(case_results, num_blades=num_blades,
                              blade_m=blade_m, tower_m=tower_m)

    return {
        "cases": case_results,
        "envelope": envelope,
        "num_cases": len(case_results),
        "num_blades": num_blades,
        "failed_cases": failed_cases,
    }


def serialize_batch_result(batch_result: Dict) -> Dict:
    """序列化批处理结果为JSON友好格式"""
    if "error" in batch_result:
        return {"error": batch_result["error"]}

    out = {
        "num_cases": batch_result["num_cases"],
        "num_blades": batch_result["num_blades"],
        "cases": [],
        "failed_cases": batch_result.get("failed_cases", []),
        "envelope": {},
    }

    # 各工况文件信息
    for cr in batch_result["cases"]:
        fi = cr.get("file_info", {})
        out["cases"].append({
            "case_id": cr.get("case_id"),
            "weight": cr.get("weight", 1.0),
            "filename": fi.get("filename"),
            "size_mb": fi.get("size_mb"),
            "condition_type": fi.get("condition_type"),
            "dlc": cr.get("dlc", "未分类"),
            "label": cr.get("label", cr.get("case_id")),
            "num_rows": fi.get("num_rows"),
            "num_cols": fi.get("num_cols"),
            "num_blades": fi.get("num_blades"),
            "sample_rate": fi.get("sample_rate"),
        })

    env = batch_result["envelope"]
    if not env:
        return out

    # 叶片校核包络
    out["envelope"]["blade"] = {}
    for bid, benv in env.get("blade_envelope", {}).items():
        out["envelope"]["blade"][bid] = {
            "extreme": _df_to_records(benv.get("extreme")),
            "fatigue": _df_to_records(benv.get("fatigue")),
        }
    # 塔顶校核包络
    tenv = env.get("tower_envelope", {})
    out["envelope"]["tower"] = {
        "extreme": _df_to_records(tenv.get("extreme")),
        "fatigue": _df_to_records(tenv.get("fatigue")),
    }
    out["envelope"]["blade_m"] = env.get("blade_m")
    out["envelope"]["tower_m"] = env.get("tower_m")
    out["envelope"]["num_cases"] = env.get("num_cases")
    out["envelope"]["global_extreme"] = _df_to_records(env.get("global_extreme"))
    out["envelope"]["global_fatigue"] = _df_to_records(env.get("global_fatigue"))

    return out


def _df_to_records(df):
    """DataFrame转records（NaN转None，兼容JSON）"""
    if df is None or df.empty:
        return []
    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
    return records
