#!/usr/bin/env python3
"""
將 SAVAR benchmark 數據轉換為 CDSD 訓練格式

使用方法:
    python convert_savar_to_cdsd.py --input savar/savar_benchmark/N4_easy/seed_0.npz \
                                     --output cdsd/dataset/savar_N4_easy_seed0
"""

import numpy as np
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, 'savar')

def convert_savar_to_cdsd(savar_file: str, output_dir: str, tau: int = 1, verbose: bool = True):
    """
    將 SAVAR .npz 文件轉換為 CDSD 需要的格式

    Args:
        savar_file: SAVAR .npz 文件路徑
        output_dir: CDSD 數據集輸出目錄
        tau: 時間滯後步數（默認=1）
        verbose: 是否打印詳細信息
    """
    # 載入 SAVAR 數據
    if verbose:
        print(f"Loading SAVAR data from: {savar_file}")
    savar_data = np.load(savar_file)

    # 提取數據
    data_field = savar_data['data_field']      # (T, spatial_points)
    latent_ts = savar_data['latent_ts']        # (T, n_latents)
    adjacency = savar_data['adjacency']        # (n_latents, n_latents)
    mode_weights = savar_data['mode_weights']  # (n_latents, grid_h, grid_w)

    T = data_field.shape[0]
    d_x = data_field.shape[1]  # 空間點數量
    d_z = latent_ts.shape[1]   # 潛變量數量

    if verbose:
        print(f"\nSAVAR Data Info:")
        print(f"  Time steps: {T}")
        print(f"  Spatial points (d_x): {d_x}")
        print(f"  Latent variables (d_z): {d_z}")
        print(f"  Mode weights shape: {mode_weights.shape}")

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 轉換 data_x: (T, d_x) → (n=1, t=T, d=1, d_x)
    data_x = data_field[np.newaxis, :, np.newaxis, :]  # (1, T, 1, d_x)
    np.save(output_path / 'data_x.npy', data_x)
    if verbose:
        print(f"\n✓ Saved data_x.npy: {data_x.shape}")

    # 2. 轉換 data_z: (T, d_z) → (n=1, t=T, d=1, d_z)
    data_z = latent_ts[np.newaxis, :, np.newaxis, :]  # (1, T, 1, d_z)
    np.save(output_path / 'data_z.npy', data_z)
    if verbose:
        print(f"✓ Saved data_z.npy: {data_z.shape}")

    # 3. 轉換 graph: (d_z, d_z) → (tau, d*d_z, d*d_z)
    #    因為 d=1，所以 d*d_z = d_z
    #    CDSD 的 graph 是時間滯後圖，包含 tau 個時間步
    graph = np.zeros((tau, d_z, d_z))
    # SAVAR 的 adjacency 是當前時刻的圖，放在第一個時間步
    graph[0] = adjacency
    np.save(output_path / 'graph.npy', graph)
    if verbose:
        print(f"✓ Saved graph.npy: {graph.shape}")
        print(f"  Adjacency matrix:\n{adjacency}")

    # 4. 轉換 graph_w (mixing matrix): (d_z, grid_h, grid_w) → (d=1, d_x, d_z)
    #    mode_weights 的形狀是 (n_latents, grid_h, grid_w)
    #    需要 reshape 成 (d=1, d_x, d_z) 並轉置
    grid_h, grid_w = mode_weights.shape[1], mode_weights.shape[2]

    # Reshape: (d_z, grid_h, grid_w) → (d_z, grid_h*grid_w)
    weights_reshaped = mode_weights.reshape(d_z, grid_h * grid_w)

    # 確保維度匹配
    if grid_h * grid_w != d_x:
        raise ValueError(f"Mode weights spatial dimension ({grid_h}x{grid_w}={grid_h*grid_w}) "
                        f"doesn't match data_field spatial points ({d_x})")

    # 轉置成 (d=1, d_x, d_z)
    graph_w = weights_reshaped.T[np.newaxis, :, :]  # (1, d_x, d_z)
    np.save(output_path / 'graph_w.npy', graph_w)
    if verbose:
        print(f"✓ Saved graph_w.npy: {graph_w.shape}")

    # 5. 創建 data_params.json
    data_params = {
        "latent": True,
        "d_x": int(d_x),
        "d_z": int(d_z),
        "tau": tau,
        "n": 1,
        "t": int(T),
        "d": 1,
        "neighborhood": 0,  # SAVAR 沒有空間鄰居概念
        "source": "SAVAR",
        "original_file": str(savar_file)
    }

    with open(output_path / 'data_params.json', 'w') as f:
        json.dump(data_params, f, indent=4)
    if verbose:
        print(f"✓ Saved data_params.json")

    # 6. 創建 best_metrics.json (用於 CDSD 訓練時的參考)
    best_metrics = {
        "note": "Placeholder metrics file for CDSD training",
        "shd": 0.0,
        "mcc": 0.0
    }

    with open(output_path / 'best_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=4)
    if verbose:
        print(f"✓ Saved best_metrics.json")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Conversion complete! CDSD dataset saved to:")
        print(f"  {output_path.absolute()}")
        print(f"\nTo train CDSD with this data:")
        print(f"  cd cdsd")
        print(f"  python main.py --config-path default_params.json \\")
        print(f"      --data-path {output_path.absolute()} \\")
        print(f"      --d-z {d_z} \\")
        print(f"      --d-x {d_x} \\")
        print(f"      --tau {tau} \\")
        print(f"      --exp-id 0")
        print(f"{'='*60}")

    return output_path


def batch_convert(input_dir: str, output_base_dir: str, tau: int = 1):
    """
    批量轉換整個 benchmark 目錄

    Args:
        input_dir: SAVAR benchmark 目錄 (例如: savar/savar_benchmark)
        output_base_dir: CDSD 數據集輸出基礎目錄
        tau: 時間滯後步數
    """
    input_path = Path(input_dir)

    # 遍歷所有配置目錄 (N4_easy, N25_med-hard, etc.)
    for config_dir in sorted(input_path.iterdir()):
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue

        print(f"\n{'='*70}")
        print(f"Processing configuration: {config_dir.name}")
        print(f"{'='*70}")

        # 遍歷所有種子文件
        for seed_file in sorted(config_dir.glob('seed_*.npz')):
            seed_name = seed_file.stem  # 'seed_0', 'seed_1', etc.
            output_dir = Path(output_base_dir) / config_dir.name / seed_name

            print(f"\nConverting {seed_file.name}...")
            convert_savar_to_cdsd(
                savar_file=str(seed_file),
                output_dir=str(output_dir),
                tau=tau,
                verbose=False
            )
            print(f"  ✓ Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SAVAR benchmark data to CDSD training format"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input SAVAR .npz file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for CDSD dataset"
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=1,
        help="Time lag parameter (default: 1)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert entire benchmark directory"
    )

    args = parser.parse_args()

    if args.batch:
        # 批量轉換模式
        if not args.input:
            args.input = "savar/savar_benchmark"
        if not args.output:
            args.output = "cdsd/dataset/savar_benchmark"

        print(f"Batch converting SAVAR benchmark...")
        print(f"  Input directory: {args.input}")
        print(f"  Output directory: {args.output}")
        batch_convert(args.input, args.output, args.tau)
    else:
        # 單文件轉換模式
        if not args.input or not args.output:
            parser.error("--input and --output are required for single file conversion")

        convert_savar_to_cdsd(args.input, args.output, args.tau, verbose=True)
