import h5py, math, numpy as np, magnum as mn
import quaternion as qt          # pip install numpy-quaternion

# ---------- 辅助转换 ---------- #
def to_vec3(arr_like):
    """
    任意 Vector3 表示 → np.float32[3]
    支持：Magnum Vector3 / ndarray / list / tuple
    """
    if isinstance(arr_like, mn.Vector3):
        return np.array([arr_like.x, arr_like.y, arr_like.z], dtype=np.float32)
    arr = np.asarray(arr_like, dtype=np.float32).reshape(3)
    return arr

def to_quat(arr_like):
    """
    任意四元数 → np.float32[4]  (w, x, y, z)
    支持：Magnum Quaternion / numpy-quaternion / list / tuple / ndarray
    """
    if isinstance(arr_like, mn.Quaternion):
        return np.array([arr_like.scalar,
                         arr_like.vector.x,
                         arr_like.vector.y,
                         arr_like.vector.z], dtype=np.float32)
    if isinstance(arr_like, qt.quaternion):
        return np.array([arr_like.w, arr_like.x, arr_like.y, arr_like.z],
                        dtype=np.float32)
    arr = np.asarray(arr_like, dtype=np.float32).reshape(4)
    # 若给成 (x,y,z,w) 可自动调整 —— 以最后一个元素绝对值最大视为 w
    if abs(arr[0]) < abs(arr[3]):           # 猜测是 [x,y,z,w]
        arr = arr[[3, 0, 1, 2]]
    return arr.astype(np.float32)

# ---------- 主保存函数 ---------- #

def save_obs_list(obs_list, h5file, sensor_key="color_0_0"):
    """
    obs_list : List[Dict[str, ndarray]]
        每个元素是一次观测的多模态 dict
    sensor_key : 选哪一个传感器写入，可换 'depth_0_0' 等
    """
    # 取出对应传感器序列
    frames = [np.asarray(obs[sensor_key]) for obs in obs_list]

    # 是否同尺寸？
    shapes = {f.shape for f in frames}
    if len(shapes) == 1:                      # 尺寸一致，可 stack 为 4-D
        dataset = np.stack(frames)            # (N,H,W,C) 或 (N,H,W)
        h5file.create_dataset(f"obs/{sensor_key}",
                              data=dataset,
                              compression="gzip")
    else:                                     # 尺寸不一样，逐帧存
        grp = h5file.create_group(f"obs/{sensor_key}")
        for i, img in enumerate(frames):
            grp.create_dataset(f"{i:06d}",
                               data=img,
                               compression="gzip")
def save_output_to_h5(output: dict, h5_path="output.h5"):
    """
    output['obs']            : List[np.ndarray(H,W,C)]
    output['follow_paths']   : List[dict]  每条 dict 结构同题目
    """
    with h5py.File(h5_path, "w") as f:
        save_obs_list(output["obs"], f, sensor_key="color_0_0")

        # ② 路径组
        grp = f.create_group("follow_paths")
        for k, fp in enumerate(output["follow_paths"]):
            g = grp.create_group(f"{k:06d}")
            g.create_dataset("obs_idx", data=np.int32(fp["obs_idx"]))
            g.create_dataset("type", data=np.int32(fp["type"]))
            dt = h5py.string_dtype(encoding='utf-8')
            g.create_dataset("desc", data=fp["desc"], dtype=dt)
  
            # follow_state
            fpos, fquat, fyaw = fp["follow_state"]
            g.create_dataset("follow_pos", data=to_vec3(fpos))
            g.create_dataset("follow_quat", data=to_quat(fquat))
            g.create_dataset("follow_yaw",  data=np.float32(fyaw))   

            # human_state
            hpos, hquat, hyaw = fp["human_state"]
            g.create_dataset("human_pos", data=to_vec3(hpos))
            g.create_dataset("human_quat", data=to_quat(hquat))
            g.create_dataset("human_yaw",  data=np.float32(hyaw))  

            # path (可变长)  → Nx7
            path_list = fp["path"]
            path_np = np.empty((len(path_list), 8), np.float32)

            for i, (pos, quat, yaw) in enumerate(path_list):
                path_np[i, :3]   = to_vec3(pos)        # 0..2  = 位置
                path_np[i, 3:7]  = to_quat(quat)       # 3..6  = 四元数 wxyz
                path_np[i, 7]    = np.float32(yaw)    
            g.create_dataset("rel_path", data=path_np, compression="gzip")

            # # path (可变长)  → Nx7
            # path_list = fp["shortest_path"]
            # path_np = np.empty((len(path_list), 3), np.float32)

            # for i, pos in enumerate(path_list):
            #     path_np[i, :3]   = to_vec3(pos)        # 0..2  = 位置

            g.create_dataset("shortest_path", data=path_np, compression="gzip")
    print(f"✅  HDF5 saved to: {h5_path}")
