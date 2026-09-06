"""
垂直轴风轮(VAWT)载荷数据处理系统 - 主入口
适配QBlade仿真输出大文件，支持极限载荷分析(EOG/NTM/EWS等所有风况) + 疲劳载荷分析

用法：
    python main.py --input <载荷文件路径> [--mode extreme|fatigue|both] [--no-panel]

示例：
    python main.py --input ../load_data_sample.txt --mode both
    python main.py --input ../load_data_sample.txt --mode extreme --no-panel
    python main.py --input ../ntm_data.txt --mode extreme  # NTM工况同样适用
"""

import os
import sys
import argparse
import logging
import time
from typing import Dict, Any

import numpy as np
import pandas as pd

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils import setup_logger, export_to_excel, timer, check_data_quality
from bigfile_io import BigFileLoader, ConditionSlicer, quick_inspect
from preprocess import ChunkPreprocessor, preprocess_dataframe
from extreme_load_analysis import run_extreme_analysis
from fatigue_analysis import run_fatigue_analysis
from blade_synthesis import synthesize_all_blades
from visualize import generate_all_plots

logger = setup_logger("VAWT_Load", config.run.log_level)


# ============================================================
# 主处理流程
# ============================================================
@timer
def run_pipeline(input_file: str, mode: str = "both",
                 include_panel: bool = True,
                 use_chunked: bool = False) -> Dict[str, Any]:
    """
    运行完整处理流程

    参数:
        input_file: QBlade仿真输出文件路径
        mode: extreme(极限载荷, 兼容旧名eog), fatigue(疲劳), both(两者)
        include_panel: 是否加载面板级分布载荷（1526列中大部分）
        use_chunked: 是否使用分块模式（超大型文件用）
    """
    # 兼容旧版模式名：eog → extreme
    if mode == "eog":
        mode = "extreme"
    start_time = time.time()
    results = {"input_file": input_file, "mode": mode}

    # ============================================================
    # 第1步：文件解析与检查
    # ============================================================
    logger.info("=" * 60)
    logger.info("第1步：文件解析")
    logger.info("=" * 60)
    print(quick_inspect(input_file))

    loader = BigFileLoader(input_file)
    info = loader.info
    mapper = loader.mapper
    results["file_info"] = {
        "filename": os.path.basename(input_file),
        "size_mb": info.file_size_mb,
        "condition_type": info.condition_type,
        "condition_name": info.condition_name,
        "num_rows": info.num_data_rows,
        "num_cols": info.num_columns,
        "num_blades": info.num_blades,
        "num_panels": info.num_panels,
    }

    # ============================================================
    # 第2步：数据加载与预处理
    # ============================================================
    logger.info("=" * 60)
    logger.info("第2步：数据加载与预处理")
    logger.info("=" * 60)

    # 获取采样率
    t_array = loader.get_time_array()
    sample_rate = info.sample_rate_hz
    logger.info(f"采样率: {sample_rate:.1f} Hz, 总时长: {info.total_time:.2f}s")

    if use_chunked and info.file_size_mb > 100:
        # 大文件分块模式：不加载全量，分块处理
        logger.info("使用分块模式处理大文件...")
        # 疲劳分析用分块模式
        if mode in ("fatigue", "both"):
            from fatigue_analysis import run_chunked_fatigue

            def channel_getter(chunk):
                blade_data = {}
                for bid in mapper.blade_total_loads:
                    bdf = mapper.get_blade_total_load_df(chunk, bid)
                    if not bdf.empty:
                        channels = {}
                        if "total_normal" in bdf.columns:
                            channels["total_normal"] = bdf["total_normal"].values
                        if "total_tangential" in bdf.columns:
                            channels["total_tangential"] = bdf["total_tangential"].values
                        if channels:
                            blade_data[bid] = channels
                return blade_data

            fatigue_results = run_chunked_fatigue(loader, mapper, channel_getter)
            results["fatigue"] = {"blade_results": fatigue_results}
        # 极限载荷分析仍需加载（瞬态工况数据量通常不大）
        if mode in ("extreme", "both"):
            df = loader.load_all(include_panel=include_panel)
            df = preprocess_dataframe(df, config.preprocess, sample_rate)
    else:
        # 常规模式：全量加载（适合<100MB文件）
        logger.info(f"全量加载 (include_panel={include_panel})...")
        df = loader.load_all(include_panel=include_panel)
        logger.info(f"原始数据: {df.shape[0]}行 x {df.shape[1]}列")

        # 数据质量检查
        quality = check_data_quality(df)
        logger.info(f"数据质量: NaN={quality['nan_count']}, Inf={quality['inf_count']}")

        # 预处理
        df = preprocess_dataframe(df, config.preprocess, sample_rate)
        logger.info(f"预处理完成: {df.shape[0]}行 x {df.shape[1]}列")
        results["data_shape"] = df.shape

    # ============================================================
    # 第3步：极限载荷分析（通用，支持EOG/NTM/EWS等所有风况）
    # ============================================================
    if mode in ("extreme", "both"):
        logger.info("=" * 60)
        logger.info(f"第3步：极限载荷分析 (工况: {info.condition_type})")
        logger.info("=" * 60)

        # 通用极值段检测（根据工况自动选择策略）
        slicer = ConditionSlicer(loader)
        cond_class = slicer.classify_condition()
        logger.info(f"工况分类: {info.condition_type} -> {cond_class}")

        extreme_segments = slicer.detect_extreme_segments()
        if extreme_segments:
            logger.info(f"检测到极值段: {[(f'{s:.2f}', f'{e:.2f}') for s, e in extreme_segments]}")
        else:
            logger.info("使用全量数据进行极值统计（稳态工况或未检测到瞬态段）")

        extreme_results = run_extreme_analysis(
            df, mapper,
            extreme_segments=extreme_segments if extreme_segments else None,
            condition_type=info.condition_type,
        )
        results["extreme"] = extreme_results
        results["eog"] = extreme_results  # 向后兼容：eog键指向extreme

        # 输出极限载荷摘要
        if not extreme_results["summary_df"].empty:
            print("\n--- 极限载荷极值汇总 ---")
            print(extreme_results["summary_df"].to_string(index=False))
        wheel = extreme_results["wheel_results"]
        if "critical_blade" in wheel:
            print(f"最不利叶片: {wheel['critical_blade']} "
                  f"(合成载荷峰值={wheel.get('critical_blade_load', 0):.1f})")

    # ============================================================
    # 第4步：疲劳载荷分析
    # ============================================================
    if mode in ("fatigue", "both") and "fatigue" not in results:
        logger.info("=" * 60)
        logger.info("第4步：疲劳载荷分析")
        logger.info("=" * 60)

        fatigue_results = run_fatigue_analysis(df, mapper, info.total_time)
        results["fatigue"] = fatigue_results

        # 输出疲劳摘要
        if not fatigue_results["summary_df"].empty:
            print("\n--- 疲劳损伤汇总 ---")
            print(fatigue_results["summary_df"].to_string(index=False))
        wf = fatigue_results["wheel_fatigue"]
        print(f"整机最大叶片损伤: {wf.get('max_blade_damage', 0):.2e}")

    # ============================================================
    # 第5步：多叶片载荷合成
    # ============================================================
    if config.synthesis.enable_synthesis and "df" in dir():
        logger.info("=" * 60)
        logger.info("第5步：多叶片载荷合成")
        logger.info("=" * 60)
        synth_results = synthesize_all_blades(df, mapper)
        results["synthesis"] = synth_results
        imb = synth_results.get("imbalance_normal", {})
        print(f"法向载荷不平衡度(均值): {imb.get('imbalance_mean', 0):.4f}")

    # ============================================================
    # 第6步：结果导出
    # ============================================================
    logger.info("=" * 60)
    logger.info("第6步：结果导出")
    logger.info("=" * 60)

    output_files = export_results(results, config)
    results["output_files"] = output_files

    # ============================================================
    # 第7步：可视化
    # ============================================================
    logger.info("=" * 60)
    logger.info("第7步：可视化")
    logger.info("=" * 60)

    if "df" in dir():
        ext_res = results.get("extreme") or results.get("eog", {})
        fat_res = results.get("fatigue", {})
        syn_res = results.get("synthesis", {})
        fig_paths = generate_all_plots(df, mapper, ext_res, fat_res, syn_res)
        results["figures"] = fig_paths

    # ============================================================
    # 完成
    # ============================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"处理完成! 总耗时: {elapsed:.2f}s")
    logger.info(f"输出目录: {config.path.output_dir}")
    logger.info("=" * 60)

    return results


# ============================================================
# 结果导出
# ============================================================
def export_results(results: Dict[str, Any], cfg) -> Dict[str, str]:
    """导出所有结果到Excel"""
    output_files = {}

    # 1. 极限载荷结果（兼容eog和extreme两个键）
    extreme = results.get("extreme") or results.get("eog")
    if extreme:
        ext_path = os.path.join(cfg.path.wheel_total_dir, "极限载荷分析结果.xlsx")
        sheets = {}
        if not extreme["summary_df"].empty:
            sheets["叶片极值汇总"] = extreme["summary_df"]
        # 整机结果
        wheel = extreme["wheel_results"]
        wheel_rows = []
        for k, v in wheel.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    wheel_rows.append({"指标": f"{k}.{k2}", "值": v2})
            else:
                wheel_rows.append({"指标": k, "值": v})
        if wheel_rows:
            sheets["整机载荷"] = pd.DataFrame(wheel_rows)
        # 各叶片详细统计
        for bid, bres in extreme["blade_results"].items():
            rows = []
            for k, v in bres.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        rows.append({"指标": f"{k}.{k2}", "值": v2})
                elif isinstance(v, pd.DataFrame):
                    continue
                else:
                    rows.append({"指标": k, "值": v})
            if rows:
                sheets[f"{bid}_详细"] = pd.DataFrame(rows)
        if sheets:
            export_to_excel(sheets, ext_path)
            output_files["extreme"] = ext_path
            output_files["eog"] = ext_path  # 向后兼容

    # 2. 疲劳结果
    if "fatigue" in results:
        fat = results["fatigue"]
        fat_path = os.path.join(cfg.path.wheel_total_dir, "疲劳分析结果.xlsx")
        sheets = {}
        if "summary_df" in fat and not fat["summary_df"].empty:
            sheets["叶片疲劳汇总"] = fat["summary_df"]
        # 各叶片载荷谱
        for bid, bres in fat.get("blade_results", {}).items():
            for ch_name, ch_data in bres.items():
                if ch_name.startswith("panel_"):
                    if "panel_fatigue_df" in ch_data:
                        sheets[f"{bid}_{ch_name}"] = ch_data["panel_fatigue_df"]
                    continue
                if "spectrum_df" in ch_data:
                    sheets[f"{bid}_{ch_name}_谱"] = ch_data["spectrum_df"]
        if sheets:
            export_to_excel(sheets, fat_path)
            output_files["fatigue"] = fat_path

    # 3. 合成结果
    if "synthesis" in results:
        syn = results["synthesis"]
        syn_path = os.path.join(cfg.path.wheel_total_dir, "叶片合成结果.xlsx")
        sheets = {}
        if "synthesis_df" in syn and not syn["synthesis_df"].empty:
            sheets["合成载荷时序"] = syn["synthesis_df"]
        for key in ["imbalance_normal", "imbalance_tangential"]:
            if key in syn:
                imb = syn[key]
                if "blade_statistics" in imb:
                    rows = []
                    for bid, stats in imb["blade_statistics"].items():
                        row = {"blade": bid}
                        row.update(stats)
                        rows.append(row)
                    sheets[key] = pd.DataFrame(rows)
        if "pulsation" in syn:
            puls = syn["pulsation"]
            rows = [{"指标": k, "值": v} for k, v in puls.items()
                    if not isinstance(v, (dict, list, np.ndarray))]
            if rows:
                sheets["旋转脉动"] = pd.DataFrame(rows)
        if sheets:
            export_to_excel(sheets, syn_path)
            output_files["synthesis"] = syn_path

    # 4. 文件信息
    if "file_info" in results:
        info_path = os.path.join(cfg.path.output_dir, "文件信息.csv")
        pd.DataFrame([results["file_info"]]).to_csv(info_path, index=False, encoding="utf-8-sig")
        output_files["file_info"] = info_path

    logger.info(f"导出 {len(output_files)} 个结果文件")
    return output_files


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="垂直轴风轮载荷数据处理系统 (QBlade适配)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --input load_data.txt --mode both
  python main.py --input load_data.txt --mode extreme --no-panel
  python main.py --input ntm_data.txt --mode extreme   # NTM工况同样适用
  python main.py --input load_data.txt --mode fatigue --chunked
        """
    )
    parser.add_argument("--input", "-i", required=True,
                        help="QBlade仿真输出文件路径 (.txt)")
    parser.add_argument("--mode", "-m", choices=["extreme", "eog", "fatigue", "both"],
                        default="both",
                        help="分析模式: extreme(极限载荷,兼容旧名eog)/fatigue/both (默认: both)")
    parser.add_argument("--no-panel", action="store_true",
                        help="不加载面板级分布载荷（减少内存，1526列->约70列）")
    parser.add_argument("--chunked", action="store_true",
                        help="使用分块模式（超大型文件>100MB推荐）")
    parser.add_argument("--chunksize", type=int, default=10000,
                        help="分块行数 (默认: 10000)")
    parser.add_argument("--output", "-o", default=None,
                        help="输出目录 (默认: ./output)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")

    args = parser.parse_args()

    # 验证输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 - {args.input}")
        sys.exit(1)

    # 更新配置
    config.run.log_level = args.log_level
    config.bigfile.chunksize = args.chunksize
    if args.output:
        config.path.output_dir = args.output
        os.makedirs(args.output, exist_ok=True)

    # 打印配置
    print(config.summary())

    # 运行
    results = run_pipeline(
        input_file=args.input,
        mode=args.mode,
        include_panel=not args.no_panel,
        use_chunked=args.chunked,
    )

    # 最终输出摘要
    print("\n" + "=" * 60)
    print("处理结果摘要")
    print("=" * 60)
    print(f"输入文件: {os.path.basename(args.input)}")
    print(f"分析模式: {args.mode}")
    if "output_files" in results:
        for k, v in results["output_files"].items():
            print(f"  {k}: {v}")
    if "figures" in results:
        print(f"  图表: {len(results['figures'])} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()
