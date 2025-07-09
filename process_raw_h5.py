#!/usr/bin/env python3
"""process_hdf5.py  ▸  Convert saved episode HDF5  ➜  PNG frames + light‑weight HDF5

兼容 **save_output_to_h5()** 中的保存格式：
    • /obs/color_0_0            or /obs 单 dataset
    • /follow_paths/000000/{
          obs_idx          (int32)
          follow_pos       (3,)
          follow_quat      (4,)
          follow_yaw       (float)
          human_pos        (3,)
          human_quat       (4,)
          human_yaw        (float)
          rel_path         (N,8)
      }

本脚本做：
1. 将 /obs/... dataset 中的每帧图像导出为 PNG (frame_000000.png …)
2. 为每条 follow_path 生成：
      observation          = PNG 路径（当前帧）
      history_observation  = 前 5 帧 PNG 路径 list
      rel_path             = 原 rel_path 但位置已减去 follow_pos.x/y
3. 把以上信息写入新的 HDF5，结构：
      /frame_paths               (N, ) string  —— 所有 PNG 路径
      /follow_paths/000000/{ observation, history_observation, rel_path, follow_* , human_* }

Usage:
    python process_hdf5.py episode.hdf5  frames_dir  episode_processed.hdf5
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import magnum as mn
import math


def wrap_pi(angle):
    """把任意角度包到 (-π, π] 区间"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def world2local(rel_path: np.ndarray,
                human_state:  np.ndarray,
                follow_quat: mn.Quaternion,
                follow_yaw: float,
                type: int) -> np.ndarray:
    """
    rel_path : (N, 8)  已经做过 位置减 follow_pos
               列序 [x y z w x y z yaw_world]
    follow_quat : mn.Quaternion  (w, xyz)  用于旋转向量
    follow_yaw  : float (rad)    已有的朝向标量
    """
    R_inv = follow_quat.inverted()          # q_f⁻¹
    out = rel_path.copy()

    human_local = R_inv.transform_vector(mn.Vector3(*human_state))
    # 1) 位置向量旋转到局部系
    for i, v in enumerate(rel_path[:, :3]):
        v_local = R_inv.transform_vector(mn.Vector3(*v))
        if type ==1:
            # out[i, :3] = [-v_local.x, v_local.y, v_local.z]
            out[i, :3] = [v_local.x, v_local.y, -v_local.z]
        else:
            out[i, :3] = [-v_local.x, v_local.y, -v_local.z]

    # 2) 四元数旋转到局部系
    for i, q in enumerate(rel_path[:, 3:7]):
        q_world = mn.Quaternion(mn.Vector3(q[1:]), q[0])
        q_local = R_inv * q_world
        out[i, 3:7] = [q_local.scalar,
                       q_local.vector.x,
                       q_local.vector.y,
                       q_local.vector.z]

    # 3) yaw 差值
    out[:, 7] = wrap_pi(rel_path[:, 7] - follow_yaw)
    if type ==1:
        # human_local =  [-human_local.x, human_local.y, human_local.z]
        human_local =  [human_local.x, human_local.y, -human_local.z]
    else:
        human_local =  [-human_local.x, human_local.y, -human_local.z]
    
    return out.astype(np.float32),human_local

def world2local_yawXZ(rel_path: np.ndarray,
                      human_state: np.ndarray,      # (3,)
                      follow_yaw: float):
    """
    rel_path     : (N,8) 已做 位置减 follow_pos
                   columns = [x y z  w x y z  yaw_world]
                   ─┬─ ─┬─ ─┬─ 水平平面 (X,Z)
                     │    │
                     height Y (保留)
    follow_yaw   : follow 朝向 (rad)，逆时针为正
    """
    c, s = math.cos(-follow_yaw), math.sin(-follow_yaw)   # 逆旋转矩阵
    out  = rel_path.astype(np.float32).copy()

    # 1) 位置 —— 只旋 (x,z)，y 不动
    x, z = out[:, 0].copy(), out[:, 2].copy()
    out[:, 0] = c*x +  s*z           # x'
    out[:, 2] = -s*x + c*z           # z'
    # 若想把 y 也对齐原点，可再做 out[:,1] -= follow_pos_y
    # out[:, 2] *= -1   
    # 2) yaw 差值                      (俯仰/滚转先忽略)
    delta_yaw = wrap_pi(rel_path[:, 7] - follow_yaw)
    out[:, 7] = delta_yaw

    # 3) 四元数 —— 仅保留 Δyaw 分量 (cos,sin)，其他置 0  → 纯平面旋转
    out[:, 3] = np.cos(delta_yaw * 0.5)   # w
    out[:, 4] = 0.0                       # x
    out[:, 5] = np.sin(delta_yaw * 0.5)   # y (旋转轴 Y)
    out[:, 6] = 0.0                       # z

    # 4) human 位置同理
    hx, hy, hz = human_state
    hxz_rot = np.array([c*hx + s*hz, -s*hx + c*hz], np.float32)
    human_local = np.array([hxz_rot[0], hy, hxz_rot[1]], np.float32)

    return out, human_local
# ----------------------------------------------------------- utilities ---- #

def save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2 or arr.shape[-1] == 1:  # 灰度
        Image.fromarray(arr.squeeze()).save(path)
    else:
        Image.fromarray(arr[..., :3]).save(path)


def locate_obs_dataset(h5: h5py.File):
    """Return (dataset, key). 支持 /obs dataset 或 /obs/color_0_0 等"""
    if "obs" not in h5:
        raise KeyError("文件里找不到 obs 组")
    node = h5["obs"]
    if isinstance(node, h5py.Dataset):
        return node, "obs"
    # group: 找 color_0_0，否则第一个
    if "color_0_0" in node:
        return node["color_0_0"], "color_0_0"
    first = next(iter(node.keys()))
    return node[first], first

def interpolate_rel_path(rel_path: np.ndarray,
                         chunk_size: int,
                         max_dist: float) -> np.ndarray:
    """
    把 (x,z,yaw) 路径插值 / 截断到固定长度 chunk_size.
    rel_path : (...,8) 或 (...,3)
    """
    if rel_path.ndim != 2 or rel_path.shape[1] not in (3, 8):
        raise ValueError("rel_path shape must be (N,3) or (N,8)")

    # 1. 取 x,z,yaw
    data = rel_path[:, [0, 2, 7]] if rel_path.shape[1] == 8 else rel_path.copy()

    # 2. 特例：全 0 或空
    if data.size == 0 or np.allclose(data, 0):
        return np.zeros((chunk_size, 3), np.float32)

    # 3. 计算沿线累积距离
    diffs  = np.diff(data[:, :2], axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    s_full = np.concatenate(([0], np.cumsum(dists)))        # len = N
    total  = s_full[-1]

    # 若超过 max_dist → 找截断点 (总长≥max_dist)
    if total > max_dist:
        idx = np.searchsorted(s_full, max_dist)
        if idx == len(s_full):
            idx -= 1
        # 截断为 idx+1 个点，并在 idx 点上插入精确 max_dist 位置
        excess = s_full[idx] - max_dist
        if excess > 1e-6 and idx > 0:
            ratio = (dists[idx-1] - excess) / dists[idx-1]
            interp_pt = data[idx-1] + ratio * (data[idx] - data[idx-1])
            data = np.vstack([data[:idx], interp_pt])
            s_full = np.concatenate(([0], np.cumsum(np.linalg.norm(
                np.diff(data[:, :2], axis=0), axis=1))))

        total = max_dist

    # 4. 等间距采样到 chunk_size
    if chunk_size == 1:
        samples = np.array([[0, 0, 0]], np.float32)
    else:
        s_samples = np.linspace(0, total, chunk_size)
        samples = np.zeros((chunk_size, 3), np.float32)
        yaw_src = np.unwrap(data[:, 2])    
        for k, s in enumerate(s_samples):
            idx = np.searchsorted(s_full, s) - 1
            idx = np.clip(idx, 0, len(s_full) - 2)
            seg_len = s_full[idx+1] - s_full[idx]
            if seg_len < 1e-8:
                samples[k] = data[idx]
            else:
                t = (s - s_full[idx]) / seg_len
                samples[k, :2] = data[idx, :2] + t * (data[idx+1, :2] - data[idx, :2])
                # yaw 带环绕，简单线插足够（前提 Δyaw 不跨 ±π）
                # samples[k, 2] = data[idx, 2] + t * (data[idx+1, 2] - data[idx, 2])
                yaw_lin = yaw_src[idx] + t * (yaw_src[idx+1] - yaw_src[idx])   # ② 线性插值
                samples[k, 2] = (yaw_lin + np.pi) % (2 * np.pi) - np.pi   

    return samples.astype(np.float32)


# def visualize_follow_path(group: h5py.Group, actions,human_local, out_png: Path):
#     """
#     两张图：
#       • 上：rel_path 在 X-Z 平面（绿色折线）+ follow/human 星标
#       • 下：Δyaw 随时间步的变化（角度 °）
#     """
#     rel_path = group["rel_path"][()]
#     # yaw_deg  = np.degrees(rel_path[:, 7])          # rad → deg
#     yaw_deg = np.degrees(actions[:2])
#     steps    = np.arange(len(rel_path))

#     fig, (ax_top, ax_bot) = plt.subplots(
#         2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [2, 1]}
#     )

#     # ── 上图：轨迹 ────────────────────────────────────────────────
#     # ax_top.plot(rel_path[:, 0], rel_path[:, 2], "g-", label="rel_path X-Z")
#     ax_top.plot(actions[:, 0], actions[:, 1], "g-", label="rel_path X-Z")
#     ax_top.scatter(0, 0, c="red", marker="*", s=100, label="follow (0,0)")
#     ax_top.scatter(human_local[0], human_local[2],
#                    c="blue", marker="*", s=100, label="human_local")
#     ax_top.set_aspect("equal"); ax_top.set_xlim(-5, 5); ax_top.set_ylim(-5, 5)
#     ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("z (m)")
#     ax_top.legend(fontsize="small")
#     ax_top.set_title(f"obs_idx = {int(group['obs_idx'][()])}")

#     # ── 下图：Δyaw-timestep ─────────────────────────────────────
#     ax_bot.plot(steps, yaw_deg, "m--", lw=1.2)
#     ax_bot.set_xlabel("timestep")
#     ax_bot.set_ylabel("Δyaw (deg)")
#     ax_bot.grid(True, alpha=0.3)

#     fig.tight_layout()
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png, dpi=140)
#     plt.close(fig)

from matplotlib.collections import LineCollection
from matplotlib import cm, colors

def visualize_follow_path(group: h5py.Group,
                          actions: np.ndarray,
                          human_local,
                          out_png: Path,
                          cmap_name: str = "viridis"):
    """
    actions : (T,3)  [x, z, yaw]   —— 已经是局部坐标
    human_local : (3,)
    """
    # ------ 数据准备 -----------------------------------------------------
    traj_xz = actions[:, :2]                       # (T,2)  x,z
    yaw_deg = np.degrees(actions[:, 2])            # (T,)   yaw°
    steps   = np.arange(len(actions))              # 0..T-1

    # 生成归一化颜色映射器
    cmap   = cm.get_cmap(cmap_name)
    norm   = colors.Normalize(vmin=0, vmax=len(actions)-1)
    colors_arr = cmap(norm(steps))

    # ------ 画布 ---------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=False
    )

    # ── 上：渐变轨迹 ───────────────────────────────────────────────
    # 把折线拆成线段集合，用 LineCollection 着色
    segs = np.concatenate(
        [traj_xz[:-1, None, :], traj_xz[1:, None, :]], axis=1
    )
    lc   = LineCollection(segs, colors=colors_arr[:-1], linewidths=2)
    ax_top.add_collection(lc)
    ax_top.scatter(0, 0, c="red", marker="*", s=100, label="follow (0,0)")
    ax_top.scatter(human_local[0], human_local[2],
                   c="blue", marker="*", s=100, label="human_local")
    ax_top.set_aspect("equal")
    ax_top.set_xlim(-5, 5); ax_top.set_ylim(-5, 5)
    ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("z (m)")
    ax_top.legend(fontsize="small")
    ax_top.set_title(f"obs_idx = {int(group['obs_idx'][()])}")

    # ── 下：Δyaw 渐变曲线 ─────────────────────────────────────────
    for i in range(len(yaw_deg)-1):
        ax_bot.plot(steps[i:i+2], yaw_deg[i:i+2],
                    color=colors_arr[i], linewidth=2)
    ax_bot.set_xlabel("timestep")
    ax_bot.set_ylabel("Δyaw (deg)")
    ax_bot.grid(True, alpha=0.3)

    # colorbar (可选)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.set_label("time step")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ----------------------------------------------------------- main --------- #

def process_one(src_file: Path, frames_root: Path, dst_root: Path, viz_root: Path|None, hist: int):
    ep_name = src_file.stem              # episode_000 etc.
    frames_dir = frames_root/ep_name
    dst_h5    = dst_root / f"{ep_name}_proc.hdf5"

    with h5py.File(src_file,"r") as fin:
        obs_ds,_ = locate_obs_dataset(fin)
        frame_paths=[]
        for i in range(len(obs_ds)):
            png = frames_dir / f"frame_{i:06d}.png"
            if not png.exists():
                save_png(obs_ds[i], png)
            rel_path = f"data/follow_data/frames/{ep_name}/frame_{i:06d}.png"
            if i ==0:
                continue
            frame_paths.append(rel_path)

        with h5py.File(dst_h5,"w") as fout:
            # cam_name = 'cam_high'
            fout.create_dataset("frame_paths",data=np.array(frame_paths,dtype=h5py.string_dtype()))
            sgrp_all=fin["follow_paths"]; dgrp_all=fout.create_group("follow_paths")
            for sub in sgrp_all:
                s=sgrp_all[sub]; d=dgrp_all.create_group(sub)
                obs_idx=int(s["obs_idx"][()])
                if obs_idx == 0:
                    continue
                type = int(s["type"][()])
                d.create_dataset("obs_idx",data=obs_idx)
                d.create_dataset(f"observations/images",data=frame_paths[obs_idx],dtype=h5py.string_dtype())
                start=max(0,obs_idx-hist);hist_paths=frame_paths[start:obs_idx]
                d.create_dataset("observations/history_images",data=np.array(hist_paths,dtype=h5py.string_dtype()))
                # for key in ("follow_pos","follow_quat","follow_yaw","human_pos","human_quat","human_yaw"):
                #     d.create_dataset(key,data=s[key][()])
                # print(obs_idx)
                # print(s["follow_yaw"][()])
                raw_path=s["rel_path"][()].astype(np.float32)
                fx,fy, fz=s["follow_pos"][()]
                human_pos = s["human_pos"][()]
                human_pos[0] -=fx; human_pos[1]-=fy; human_pos[2]-=fz
                follow_quat =  mn.Quaternion(mn.Vector3(s["follow_quat"][()][1:]), s["follow_quat"][()][0])
                follow_yaw  = s["follow_yaw"][()]
                raw_path[:,0]-=fx; raw_path[:,1]-=fy; raw_path[:,2]-=fz
                reletive_path, huamn_local  = world2local(raw_path,human_pos, follow_quat, follow_yaw,type)
                # reletive_path, huamn_local  = world2local_yawXZ(raw_path,human_pos, follow_yaw)
                d.create_dataset("rel_path",data=reletive_path)
                actions = interpolate_rel_path(reletive_path, 30, 3.0)
                d.create_dataset('language_raw', data="follow the human")
                d.create_dataset('action', data=actions, compression='gzip')
                qposes = np.zeros_like(actions)
                d.create_dataset('qpos', data=qposes, compression='gzip')
                # if viz_root:
                #     visualize_follow_path(d, actions,huamn_local, viz_root/ep_name/f"action_{obs_idx}_{type}.png")
    print(f"✓ {ep_name} -> {dst_h5}")

def main(src_dir: Path, frames_dir: Path, dst_dir: Path, viz_dir: Path|None, history:int):
    src_dir, frames_dir, dst_dir = map(Path,(src_dir,frames_dir,dst_dir))
    frames_dir.mkdir(parents=True,exist_ok=True)
    dst_dir.mkdir(parents=True,exist_ok=True)

    h5_files=sorted(src_dir.glob("*.hdf5"))
    if not h5_files:
        print("‼ 未找到 *.hdf5 文件于",src_dir); return

    for f in h5_files:
        process_one(f, frames_dir, dst_dir, viz_dir, history)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="batch convert episodes")
    ap.add_argument("src_dir", help="包含多个 episode.hdf5 的目录")
    ap.add_argument("frames_dir", help="PNG 根目录")
    ap.add_argument("dst_dir", help="processed h5 根目录")
    ap.add_argument("--viz", help="可视化输出根目录")
    ap.add_argument("--history",type=int,default=5)
    args=ap.parse_args()

    viz=Path(args.viz) if args.viz else None
    main(Path(args.src_dir), Path(args.frames_dir), Path(args.dst_dir), viz, args.history)

