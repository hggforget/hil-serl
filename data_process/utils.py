import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
import time
# import torch
import glob
import pickle as pkl
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R
#import open3d as o3d
import fnmatch
# import torchvision.transforms as transforms
from scipy import signal
import copy
import sys

def decompress_images(compressed_data_array):
    """
    批量解压缩图像数据
    Args:
        compressed_data_array: numpy数组，形状为(batch_size, width, height, 3)
                             或单个压缩图像数据
    Returns:
        numpy.ndarray: 解压后的图像数组，形状为(batch_size, width, height, 3)
                      或单个解压后的图像
    """
    # 检查输入是否为numpy数组
    if not isinstance(compressed_data_array, np.ndarray):
        compressed_data_array = np.array([compressed_data_array])

    # 如果输入已经是RGB图像格式，直接返回
    if len(compressed_data_array.shape) == 4 and compressed_data_array.shape[-1] == 3:
        return compressed_data_array
    
    # 获取batch大小
    batch_size = compressed_data_array.shape[0]
    
    # 创建空列表存储解压后的图像
    decompressed_images = []
    
    # 遍历每个压缩图像
    for i in range(batch_size):
        try:
            # 将压缩的字节数据转换为numpy数组
            np_data = np.frombuffer(compressed_data_array[i], np.uint8)
            
            # 使用OpenCV解压缩图像
            image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"无法解压缩第{i}个图像数据")
            
            # 转换图像为uint8类型
            image_uint8 = image.astype(np.uint8)

            decompressed_images.append(image_uint8)
            
        except Exception as e:
            print(f"解压缩第{i}个图像失败: {str(e)}")
            # 在失败时添加一个空图像或上一帧的图像
            if decompressed_images:
                decompressed_images.append(decompressed_images[-1])
            else:
                # 如果是第一帧就失败，创建一个黑色图像
                decompressed_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    # 转换为numpy数组
    decompressed_array = np.stack(decompressed_images, axis=0)
    
    # 如果输入是单个图像，返���单个图像而不是数组
    if batch_size == 1:
        return decompressed_array[0]
    
    return decompressed_array
def compress_image(img, quality=80):
    """
    压缩RGB图像
    Args:
        img: RGB图像 (numpy array)
        quality: JPEG压缩质量(1-100)，默认80
    Returns:
        compressed_data: 压缩后的字节数据
        decompressed_img: 解压后的图像
    """
    # 如果输入已经是字节数据，直接返回
    if isinstance(img, bytes):
        return img

    # 检查输入
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Invalid input image")
        
    # JPEG压缩
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    compressed_data = encoded.tobytes()
    
    # 解压缩回图像
    # decoded_img = cv2.imdecode(
    #     np.frombuffer(compressed_data, np.uint8), 
    #     cv2.IMREAD_COLOR
    # )
    
    return compressed_data

def decompress_image(compressed_data):
    # 将压缩的字节数据转换为 numpy 数组
    np_data = np.frombuffer(compressed_data, np.uint8)
    
    # 使用 OpenCV 解压缩图像
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("无法解压缩图像数据")
    
    # 转换图像为 uint8 类型并返回
    image_uint8 = image.astype(np.uint8)
    
    return image_uint8

def find_all_hdf5(dataset_dir):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def load_hdf5(dataset_dir, dataset_name):

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    def _recurisive_create(root, index='/'):
        if isinstance(root[index], h5py.Group):
            return {key: _recurisive_create(root, index + '/' + key) for key in root[index].keys()}
        elif isinstance(root[index], h5py.Dataset):
            return root[index][()]
    episode = {}
    with h5py.File(dataset_path, 'r') as root:
        episode = _recurisive_create(root)
    return episode




# 角度范围：[0, π]
# 轴方向：任意方向（自由调整以保持角度在[0, π]范围内）
def normalize_axis_angle(axis_angle, eps=1e-8):
    """标准化轴角表示
    
    Args:
        axis_angle: 输入的轴角向量 [rx, ry, rz]
        
    Returns:
        normalized: 标准化后的轴角向量，角度范围在[0, π]
    """
    angle = np.linalg.norm(axis_angle)
    if angle < eps:
        return np.zeros(3)
    #print("angle:", angle)
    # 如果角度大于π，将其映射到[0, π]范围
    if angle > np.pi:
        # 取模运算，保持在[-π, π]范围内
        reduced_angle = angle % (2 * np.pi)
        if reduced_angle > np.pi:
            reduced_angle = 2 * np.pi - reduced_angle
            axis = -axis_angle / angle
        else:
            axis = axis_angle / angle
        return axis * reduced_angle



    # 处理接近π的情况（180度旋转）
    if np.isclose(angle, np.pi, atol=eps):
        # 选择一个标准的表示方式
        # 例如：确保第一个非零分量为正
        axis = standardize_axis(axis)

    return axis_angle

def standardize_axis(axis, eps=1e-8):
    """标准化旋转轴的表示
    
    用于180度旋转时确保轴的表示唯一
    """
    for i in range(3):
        if abs(axis[i]) > eps:
            if axis[i] < 0:
                return -axis
            return axis
    return np.array([1.0, 0.0, 0.0])  # 默认选择x轴作为标准轴

def rotation_matrix_to_axis_angle(R, eps=1e-8):
    """将3x3旋转矩阵转换为轴角表示
    
    Args:
        R (numpy.ndarray): 3x3旋转矩阵
        eps (float): 数值计算的容差值
        
    Returns:
        numpy.ndarray: 轴角表示 [rx, ry, rz]，其中向量的方向表示旋转轴，模长表示旋转角度(弧度制)
        
    Raises:
        ValueError: 如果输入不是有效的旋转矩阵
    """
    # 输入检查
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError(f"Input must be 3x3 matrix, got shape {R.shape}")
    
    # 验证是否为有效的旋转矩阵
    if not np.allclose(np.dot(R, R.T), np.eye(3), rtol=eps):
        raise ValueError("Matrix is not orthogonal")
    if not np.isclose(np.linalg.det(R), 1.0, rtol=eps):
        raise ValueError("Matrix determinant is not 1")

    # 计算旋转角度
    # 使用 (trace(R) - 1) / 2 = cos(theta)
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值稳定性
    theta = np.arccos(cos_theta)

    # 特殊情况处理
    if np.isclose(theta, 0, atol=eps):  # 无旋转
        return np.zeros(3)
        
    if np.isclose(theta, np.pi, atol=eps):  # 180度旋转
        # 寻找最大的对角线元素
        diag = np.diag(R)
        k = np.argmax(diag)
        if diag[k] < -1 + eps:  # 数值稳定性检查
            raise ValueError("Invalid rotation matrix")
            
        axis = np.zeros(3)
        # 计算第k列的平方根
        axis[k] = np.sqrt((diag[k] + 1) / 2)
        # 计算非对角线元素
        if k == 0:
            axis[1] = R[0,1] / (2 * axis[0])
            axis[2] = R[0,2] / (2 * axis[0])
        elif k == 1:
            axis[0] = R[0,1] / (2 * axis[1])
            axis[2] = R[1,2] / (2 * axis[1])
        else:
            axis[0] = R[0,2] / (2 * axis[2])
            axis[1] = R[1,2] / (2 * axis[2])
            
        return axis * np.pi
    
    # 标准情况：使用反对称矩阵提取旋转轴
    


    if np.sin(theta) > 1e-6:# 防止除以0
        axis = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
            ]) / (2 * np.sin(theta))
    else:
        # 处理 sinTheta 接近 0 的情况，这里假设旋转角接近 0
        axisX = 1
        axisY = 0
        axisZ = 0
        axis = np.array([axisX, axisY, axisZ])

    return normalize_axis_angle(axis * theta)

def decompress_image(compressed_data):
    # 将压缩的字节数据转换为 numpy 数组
    np_data = np.frombuffer(compressed_data, np.uint8)
    
    # 使用 OpenCV 解压缩图像
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("无法解压缩图像数据")
    
    # 转换图像为 uint8 类型并返回
    image_uint8 = image.astype(np.uint8)
    
    return image_uint8


def quaternion_to_rotation_matrix(quaternion, eps=1e-8):
    """将四元数转换为3x3旋转矩阵
    
    Args:
        quaternion (numpy.ndarray): 四元数 [x, y, z, w]，可以是非单位四元数
        eps (float, optional): 数值计算的容差值。默认为1e-8
        
    Returns:
        numpy.ndarray: 3x3旋转矩阵
        
    Raises:
        ValueError: 如果输入的四元数格式不正确
        
    Notes:
        - 输入四元数应遵循[x,y,z,w]顺序
        - 返回的矩阵是正交矩阵，满足 R * R.T = I
        - 行列式为+1，确保是旋转而不是反射
    """
    # 输入检查
    if not isinstance(quaternion, np.ndarray):
        quaternion = np.array(quaternion, dtype=np.float64)
    
    if quaternion.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {quaternion.shape}")
    
    # 提取四元数分量并归一化
    norm = np.sqrt(np.sum(quaternion**2))
    if norm < eps:
        raise ValueError("Quaternion magnitude is too close to zero")
    
    x, y, z, w = quaternion / norm
    #print("x, y, z, w:", x, y, z, w)
    
    # 预计算一些常用项
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # 构建旋转矩阵
    R = np.array([
        [1 - 2*(y2 + z2),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(x2 + z2),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(x2 + y2)]
    ], dtype=np.float64)
    
    # 数值稳定性检查
    if not np.allclose(np.dot(R, R.T), np.eye(3), rtol=eps, atol=eps):
        raise ValueError("Resulting matrix is not orthogonal")
    
    if abs(np.linalg.det(R) - 1.0) > eps:
        raise ValueError("Resulting matrix is not a proper rotation matrix")
    
    return R

def axis_angle_to_quaternion(axis_angle):
    """
    将轴角向量转换为四元数
    
    参数:
        axis_angle: numpy数组，表示轴角向量（方向表示旋转轴，模长表示旋转角度）
    
    返回:
        quaternion: numpy数组，格式为[x, y, z, w]
    """
    # 计算旋转角度（轴角向量的模长）
    angle = np.linalg.norm(axis_angle)
    #print("angle:", angle)
    
    # 如果角度接近0，返回单位四元数
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    # 提取旋转轴（归一化轴角向量）
    axis = axis_angle / angle
    
    # 计算四元数分量
    sin_theta_2 = np.sin(angle / 2)
    cos_theta_2 = np.cos(angle / 2)
    
    x = axis[0] * sin_theta_2
    y = axis[1] * sin_theta_2
    z = axis[2] * sin_theta_2
    w = cos_theta_2
    
    return np.array([x, y, z, w])

def relative_pose(pose1, pose2):
    #[x, y, z, qx, qy, qz, qw]
    # 提取位置和四元数              
    pos1 = np.array(pose1[:3])
    pos2 = np.array(pose2[:3])
    quat1 = np.array(pose1[3:])
    quat2 = np.array(pose2[3:])

    # 计算相对位置
    relative_position = pos2 - pos1
    
    # 计算相对旋转
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    
    # 相对旋转是 r1 的逆乘以 r2
    relative_rotation = r1.inv() * r2
    
    relative_quat = relative_rotation.as_quat()
    #print("relative_quat:", relative_quat)
    # 将四元数转换为轴角
    rotation_matrix = quaternion_to_rotation_matrix(relative_quat)
    axis_angle = rotation_matrix_to_axis_angle(rotation_matrix)

    # 返回相对位姿
    #[x, y, z, a1, a2, a3]
    return np.concatenate((relative_position, axis_angle))

def absolute_pose(initial_pose, relative_pose):
    """计算绝对位姿
    
    Args:
        initial_pose: 初始位姿 [x, y, z, qx, qy, qz, qw]
        relative_pose: 相对位姿 [x, y, z, a1,a2,a3]
        
    Returns:
        absolute_pose_array: 绝对位姿 [x, y, z, qx, qy, qz, qw]
    """
    # 提取位置和四元数
    pos_i, quat_i = initial_pose[:3], initial_pose[3:]
    rel_pos, rel_axisangle = relative_pose[:3], relative_pose[3:]

    rel_quat = axis_angle_to_quaternion(rel_axisangle)
    
    # 计算绝对位置
    absolute_position = pos_i + rel_pos
    
    # 计算绝对旋转（四元数自动归一化）
    r_absolute = R.from_quat(quat_i) * R.from_quat(rel_quat)
    absolute_quaternion = r_absolute.as_quat()
    
    # 合并位置和四元数
    return np.concatenate([absolute_position, absolute_quaternion])



def annotate(images_array):
    # 获取图像数组的形状信息
    num_frames, height, width, channels = images_array.shape
    index = 0  # 初始化当前帧的索引为0
    step = 1  # 初始化每次跳过的帧数为1
    rewards = np.zeros((num_frames,), dtype=np.float32)  # 创建一个全0的数组，用于记录每一帧的奖励值
    auto_play = False  # 初始模式为手动播放
    
    while True:
        print(f'Frame : {index}')  # 打印当前帧的索引
        # 获取当前帧的图像数据
        frame = images_array[index]

        # 如果图像是 RGB 格式，转换为 BGR 格式以便 OpenCV 显示
        if channels == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 显示当前帧的图像
        cv2.imshow('Frame Viewer', frame)

        # 等待按键事件，根据 auto_play 的值设置等待时间
        delay = 1 if auto_play else 0
        key = cv2.waitKey(delay) & 0xFF  # 使用 & 0xFF 以兼容不同平台

        # 按回车键切换播放模式
        if key == ord('\r') or key == ord('\n'):  # 处理不同平台的回车键
            auto_play = not auto_play  # 切换播放模式
        # 按 'ESC' 键退出
        elif key == 27:  # ESC 键的 ASCII 码
            break
        # 按 '+' 键增加每次跳过的帧数
        elif key == 43:
            step = min(10, step + 1)
            print(f'Current Step: {step}')
        # 按 '-' 键减少每次跳过的帧数
        elif key == 45:
            step = max(0, step - 1)
            print(f'Current Step: {step}')
        # 按 'r' 键重置每次跳过的帧数为1
        elif key == 114:
            step = 1
            print(f'Current Step: {step}')
            
        if not auto_play:
            # 按右箭头键显示下一个帧
            if key == 83:  # 右箭头键的 ASCII 码
                index = (index + step) % num_frames  # 循环显示
            # 按左箭头键显示上一个帧
            elif key == 81:  # 左箭头键的 ASCII 码
                index = (index - step) % num_frames  # 循环显示
            # 按上箭头键标记奖励并显示下一个帧
            elif key == 82:  # 上箭头键的 ASCII 码
                rewards[index:] = 1.0
                print(f'annotate frame {index} positive')
            # 按下箭头键标记无奖励并显示下一个帧
            elif key == 84:  # 下箭头键的 ASCII 码
                rewards[index:] = 0.0
                print(f'annotate frame {index} negtive')
            # 按 'p' 键打印当前帧的奖励值
            elif key == 112:
                print(f'annotate frame {index} reward={rewards[index]}')
   
        if auto_play:
            index = (index + step) % num_frames  # 自动播放时，自动跳到下一帧
      
    return rewards if input(f'confirm? y/n').lower() == 'y' else annotate(images_array)  # 返回每一帧的奖励值

def create_demos(env, episodes):
    demos = []
    for episode in tqdm(episodes):
        episode_len = len(episode)
        for frame_index in range(episode_len):
            observation_data = dict(
                state=env.process_state({key: getattr(episode, key)[frame_index] for key in env.proprio_space.spaces.keys()}),
                **env.process_images({key: getattr(episode, key)[frame_index] for key in env.cameras_config.keys()}),
            )
            next_observation_data = dict(
                state=env.process_state({key: getattr(episode, key)[frame_index + 1] for key in env.proprio_space.spaces.keys()}),
                **env.process_images({key: getattr(episode, key)[frame_index + 1] for key in env.cameras_config.keys()}),
            )
            actions = env.reverse_process_action({key: episode.action[key][frame_index] for key in env.action_dict_space.spaces.keys()})
            rewards = np.array(episode.step_rewards[frame_index], dtype=np.float32)
            dataset_dict = dict(
            observations=observation_data, 
            next_observations=next_observation_data,
            actions=actions,
            rewards=rewards,
            masks=np.array(1 - episode.step_rewards[frame_index], dtype=np.float32),
            dones=rewards,
            )
            demos.append(dataset_dict)
    return demos