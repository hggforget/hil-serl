#coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
import time
# import torch
import glob
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R
#import open3d as o3d
import fnmatch
# import torchvision.transforms as transforms
from scipy import signal
import copy


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

def compress_depth_image(depth_img, quality=100):
    """
    压缩深度图
    Args:
        depth_img: 深度图 (numpy array, dtype=float32/uint16)
        quality: JPEG压缩质量(1-100)
    Returns:
        compressed_data: 压缩后的字节数据
    """
    # 如果输入已经是字节数据，直接返回
    if isinstance(depth_img, bytes):
        return depth_img

    # 检查输入
    if depth_img is None or not isinstance(depth_img, np.ndarray):
        raise ValueError("Invalid input depth image")
    
    # 将深度图转换为uint16
    if depth_img.dtype != np.uint16:
        depth_img = depth_img.astype(np.uint16)
    
    # 分离高8位和低8位
    high = (depth_img >> 8).astype(np.uint8)
    low = (depth_img & 0xFF).astype(np.uint8)
    
    # 将两个通道合并为一个RGB像
    combined = np.dstack((high, low, np.zeros_like(high)))
    
    # JPEG压缩
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', combined, encode_param)
    compressed_data = encoded.tobytes()
    
    return compressed_data


def decompress_depth_images(compressed_data, height=480, width=640):
    """
    批量解压缩深度图数据
    Args:
        compressed_data: 深度图字节数据
        height: 深度图高度
        width: 深度图宽度
    Returns:
        numpy.ndarray: 解压后的深度图数组
    """
    if not isinstance(compressed_data, (list, np.ndarray)):
        compressed_data = [compressed_data]
    
    if isinstance(compressed_data, np.ndarray):
        if len(compressed_data.shape) == 3:
            return compressed_data
        elif len(compressed_data.shape) == 2:
            return compressed_data
    
    depth_images = []
    for i, data in enumerate(compressed_data):
        try:
            # 打印字节数据的大小，帮助调试
            #print(f"字节数据大小: {len(data)} bytes")
            expected_size = height * width * 2  # uint16 = 2 bytes
            #print(f"预期大小: {expected_size} bytes")
            
            # 检查数据是否需要先解码
            if len(data) != expected_size:
                # 尝试先用cv2解码
                np_data = np.frombuffer(data, np.uint8)
                decoded = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
                if decoded is not None:
                    depth_images.append(decoded)
                    continue
                else:
                    raise ValueError(f"数据大小不匹配且无法解码: 实际 {len(data)}, 预期 {expected_size}")
            
            # 直接转换为uint16数组
            depth_array = np.frombuffer(data, dtype=np.uint16)
            depth_image = depth_array.reshape(height, width)
            depth_images.append(depth_image)
            
        except Exception as e:
            print(f"处理第 {i} 个深度图时出错: {str(e)}")
            # 在失败时添加一个空深度图或上一帧的深度图
            if depth_images:
                depth_images.append(depth_images[-1])
            else:
                depth_images.append(np.zeros((height, width), dtype=np.uint16))
    
    depth_images = np.stack(depth_images, axis=0)
    
    return depth_images[0] if depth_images.shape[0] == 1 else depth_images





def numpy_images_to_video(image_arrays, output_video_original, fps=30, absolute_path=None):
    # 确保至少有一个图像数组
    if image_arrays.size == 0:
        print("No images provided.")
        return
    if absolute_path is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        output_video = script_path + output_video
    else:
        output_video = absolute_path
    # 获取图像的宽度和高度
    height, width, layers = image_arrays[0].shape

    # for i, image in enumerate(image_arrays):
    #     print(output_video_original)
    #     image_path = output_video_original + f"_frame_{i}.jpg"
    #     cv2.imwrite(image_path, image)
    #     print(f"存储第 {i} 张图片: {image_path}")

    # 定义视频编码和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID' 等其他编码格式
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in image_arrays:
        # if image.shape[2] == 3:  # 确保是三通道图像
        #     bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 从 RGB 转换为 BGR
        # else:
        #     bgr_image = image  # 如果不是三通道图像，直接使用
        video.write(image)

    # 释放 video 对象
    video.release()
    print(f"Video saved as {output_video}")

def arm_position_plot(data,arm_name,absolute_path = None):
    # 提取位置数据
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    
    
    ##############################画曲线#########################################
    
    # 创建时间轴（假设每个样本间隔为1单位时间）
    time = np.arange(data.shape[0])
    
    # 绘制每个手指的关节角度随时间变化的图像
    plt.figure(figsize=(10, 6))


    plt.plot(time, x, label='x')
    plt.plot(time, y, label='y')
    plt.plot(time, z, label='z')

    # 设置图形标题和标签
    plt.title(arm_name)
    plt.xlabel('Time')
    plt.ylabel('Arm Position')
    plt.legend()

    
    
    ##############################画坐标系#########################################
    
    # # 创建一个 3D 图形
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制轨迹
    # ax.plot(x, y, z, label=arm_name, color='b')

    # # 设置标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title(arm_name)

    # # 显示图例
    # ax.legend()

    # 保存图像
    if absolute_path is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        arm_name_path = script_path+'/output_dataprocess/'+arm_name + '.png'
    else:
        arm_name_path = absolute_path + '/' + arm_name + '.png'
    plt.savefig(arm_name_path)

    print(f"{arm_name} saved as {arm_name_path}")
    plt.close()

def hand_angle_plot(data,hand_name,absolute_path=None):

    # 创建时间轴（假设每个样本间隔为1单位时间）
    time = np.arange(data.shape[0])

    # 绘制每个手指的关节角度随时间变化的图像
    plt.figure(figsize=(10, 6))

    for i in range(data.shape[1]):
        plt.plot(time, data[:, i], label=f'Finger {i+1}')

    # 设置图形标题和标签
    plt.title(hand_name)
    plt.xlabel('Time')
    plt.ylabel('Joint Angle (degrees)')
    plt.legend()

    # 保存图像
    if absolute_path is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        hand_name_path = script_path + '/output_dataprocess/'+hand_name+".png"
    else:
        hand_name_path  = absolute_path + '/' + hand_name + '.png'
    plt.savefig(hand_name_path)

    # # 显示图形
    #plt.show()

    print(f"{hand_name} saved as {hand_name_path}")
    plt.close()

def save_byte_string_array_to_file(byte_strings, name, absolute_path = None):
    """
    将字节字符串数组转换为普通字符串并存储到文件中。

    :param byte_strings: 要存储的字节字符串数组（列表或一维NumPy数组）。
    :param filename: 保存文件的名称。
    """
    # 将字节字符串转换为普通字符串
    strings = [str(i)+": "+b.decode('utf-8') for i,b in enumerate(byte_strings)]
    
    # 转换为NumPy数组
    strings_array = np.array(strings)
    
    if absolute_path is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        name_path = script_path + '/output_dataprocess/'+name+".txt"
    else:
        name_path = absolute_path + '/' + name + '.txt'

    
    # 使用numpy.savetxt将字符串数组保存到文本文件中
    np.savetxt(name_path, strings_array, fmt='%s', delimiter='\n')

def save_depth_maps_as_video(depth_maps, output_file, fps=30, absolute_path = None):
    """
    将深度图批量保存为视频文件。

    :param depth_maps: 三维NumPy数组，形状为(num_frames, height, width)，表示一批深度图。
    :param output_file: 输出视频文件路径，例如 'output.avi'。
    :param fps: 视频帧率。
    """
    num_frames, height, width = depth_maps.shape
    if absolute_path is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        output_file = script_path + output_file
    else:
        output_file = absolute_path

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)

    for i in range(num_frames):
        # 将深度图转换为8位图像
        depth_image = cv2.normalize(depth_maps[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 写入帧
        video_writer.write(depth_image)

    video_writer.release()
    
    
    
    print(f"Video saved as {output_file}")

def render_time_series_point_clouds_to_video(point_clouds, output_file='time_series_point_clouds.avi', fps=30):
    """
    渲染时间序列点云并保存为视频文件。

    :param point_clouds: 点云数组，形状为 (steps, num, 3)。
    :param output_file: 输出视频文件路径。
    :param fps: 视频帧率。
    """
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    
    script_path = os.path.dirname(os.path.abspath(__file__))
    output_file = script_path + output_file
    
    # 获取窗口尺寸
    width, height = 640, 480
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().point_size = 2

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for step in range(point_clouds.shape[0]):
        # 清除上一帧的几何体
        vis.clear_geometries()
        
        # 创建新的点云对象并添加到可视化
        #points = point_clouds[step]
        points = point_clouds[step].astype(np.float64)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        
        vis.add_geometry(point_cloud)

        # 渲染并捕获图像
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)

        # 写入视频帧
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    vis.destroy_window()
    video_writer.release()
    print(f"Point cloud  saved as {output_file}")


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files



def load_hdf5(args, dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        

        ## 本体观测信息
        obs_arm_left = root['/observations/arm_left'][()]
        obs_arm_right = root['/observations/arm_right'][()]

        obs_hand_left = root['/observations/hand_left'][()]
        obs_hand_right = root['/observations/hand_right'][()]


        if args.use_hand_force_process:
            obs_hand_left_other = obs_hand_left[:,:12]
            obs_hand_left_force = obs_hand_left[:,12:]


            obs_hand_right_other = obs_hand_right[:,:12]
            obs_hand_right_force = obs_hand_right[:,12:]

            # 处理力传感器数据，将大于32767的值减去65535
            obs_hand_left_force = np.where(obs_hand_left_force > 32767, 
                                          obs_hand_left_force - 65535, 
                                          obs_hand_left_force)
            obs_hand_right_force = np.where(obs_hand_right_force > 32767, 
                                          obs_hand_right_force - 65535, 
                                          obs_hand_right_force)

            obs_hand_left = np.concatenate((obs_hand_left_other,obs_hand_left_force),axis=1)
            obs_hand_right = np.concatenate((obs_hand_right_other,obs_hand_right_force),axis=1)




        ## 图像信息
        image_dict = dict()
        image_dict_depth = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        
        if args.use_depth_image:
            for cam_name in args.camera_names_depth:
                image_dict_depth[cam_name] = root[f'/observations/images_depth/{cam_name}'][()]



        if args.use_pointcloud:
        ## 点云信息
            point_cloud = root['/observations/pointcloud'][()]
        else:
            point_cloud = None


        ## 状态
        state = root['/observations/state'][()]


        ## 动作信息
        action_arm_left = root['/action/arm_left'][()]
        action_arm_right = root['/action/arm_right'][()]

        action_hand_left = root['/action/hand_left'][()]
        action_hand_right = root['/action/hand_right'][()]


        base_action = root['/base_action'][()]


        ## 奖励
        reward = root['/reward'][()]
        
        ## step reward
        # if 'step_rewards' in root.keys():
        #     step_reward = root['/step_rewards'][()]
        # else:
        #     step_reward = None
        

        

    return obs_arm_left, obs_arm_right, obs_hand_left, obs_hand_right, image_dict, image_dict_depth, point_cloud, \
        state, action_arm_left, action_arm_right, action_hand_left, action_hand_right, base_action, reward


# 保存数据函数
def save_data(args, dataset_path, obs_arm_left, obs_arm_right, obs_hand_left, obs_hand_right, image_dict, image_dict_depth, point_cloud, 
                  state, action_arm_left, action_arm_right, action_hand_left, action_hand_right, base_action, reward,save_dataset_dir,episode_idx,
                  phase_name, hqsam_act_body, hqsam_act_head , hqsam_act_body_robot, hqsam_act_head_robot, transformations_augment):
    
    
    if args.use_smooth_hand_data:
        obs_hand_left = smooth_hand_data(obs_hand_left)
        obs_hand_right = smooth_hand_data(obs_hand_right)
    
    
    # 数据字典
    data_size = len(action_arm_right)
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/arm_left': [],
        '/observations/arm_right': [],
        '/observations/hand_left': [],
        '/observations/hand_right': [],
        '/observations/relative_arm_left': [],
        '/observations/relative_arm_right': [],
        '/observations/relative_hand_left': [],
        '/observations/relative_hand_right': [],
        '/observations/instruction': [],
        '/action/arm_left': [],
        '/action/arm_right': [],
        '/action/hand_left': [],
        '/action/hand_right': [],
        '/action/relative_arm_left': [],
        '/action/relative_arm_right': [],
        '/action/relative_hand_left': [],
        '/action/relative_hand_right': [],
        '/base_action': [],
        '/reward': [],
    }

    ## reward
    data_dict['/reward'].append(reward)
    
    
    # 相机字典  观察的图像
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_augment:
            data_dict[f'/observations/images/{cam_name}_augment'] = []
            data_dict[f'/observations/images/{cam_name}_mask'] = []
    if args.use_depth_image:
        for cam_name in args.camera_names_depth:
            data_dict[f'/observations/images_depth/{cam_name}'] = []
    if args.use_pointcloud:
        data_dict[f'/observations/pointcloud'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    robot_tracking_step = 0

    for i in range(data_size):

        # 往字典里面添值
        # 先将bytes解码为字符串，然后去除"hand teleoperation ongoing"，最后重新编码为bytes
        if "hand teleoperation ongoing " in state[i].decode('utf-8'):
            cleaned_instruction = state[i].decode('utf-8').replace("hand teleoperation ongoing ", "").strip().encode('utf-8')
        elif "hand teleoperation ongoing" in state[i].decode('utf-8'):
            cleaned_instruction = state[i].decode('utf-8').replace("hand teleoperation ongoing", "").strip().encode('utf-8')
        elif "arm planning ongoing" in state[i].decode('utf-8'):
            cleaned_instruction = state[i].decode('utf-8').replace("arm planning ongoing", "").strip().encode('utf-8')
        elif "arm planning ongoing " in state[i].decode('utf-8'):
            cleaned_instruction = state[i].decode('utf-8').replace("arm planning ongoing ", "").strip().encode('utf-8')
        else:
            cleaned_instruction = state[i]
        # import ipdb;ipdb.set_trace();
        #print(cleaned_instruction.decode('utf-8'))  # 用于调试，打印人类可读的字符串
        data_dict['/observations/instruction'].append(cleaned_instruction)

        # 观测
        data_dict['/observations/arm_left'].append(obs_arm_left[i])
        data_dict['/observations/arm_right'].append(obs_arm_right[i])
        data_dict['/observations/hand_left'].append(obs_hand_left[i])
        data_dict['/observations/hand_right'].append(obs_hand_right[i])


        # 相对观测
        relative_obs_arm_left_term = relative_pose(obs_arm_left[0][6:],obs_arm_left[i][6:])
        data_dict['/observations/relative_arm_left'].append(relative_obs_arm_left_term)
        relative_obs_arm_right_term = relative_pose(obs_arm_right[0][6:],obs_arm_right[i][6:])
        data_dict['/observations/relative_arm_right'].append(relative_obs_arm_right_term)


        relative_left_hand_feedback_angle = obs_hand_left[i][:6]
        relative_left_hand_feedback_force = obs_hand_left[i][12:] - obs_hand_left[0][12:]
        relative_left_hand_feedback = np.concatenate((relative_left_hand_feedback_angle,relative_left_hand_feedback_force))

        relative_right_hand_feedback_angle = obs_hand_right[i][:6]
        relative_right_hand_feedback_force = obs_hand_right[i][12:]- obs_hand_right[0][12:]
        relative_right_hand_feedback = np.concatenate((relative_right_hand_feedback_angle,relative_right_hand_feedback_force))

        data_dict['/observations/relative_hand_left'].append(relative_left_hand_feedback)
        data_dict['/observations/relative_hand_right'].append(relative_right_hand_feedback)
        

        # 实际发的action
        data_dict['/action/arm_left'].append(action_arm_left[i])
        data_dict['/action/arm_right'].append(action_arm_right[i])
        data_dict['/action/hand_left'].append(action_hand_left[i])
        data_dict['/action/hand_right'].append(action_hand_right[i])
        data_dict['/base_action'].append(base_action[i])



        # 相对动作
        #relative_action_arm_left_term = relative_pose(action_arm_left[0],action_arm_left[i])
        #relative_action_arm_right_term = relative_pose(action_arm_right[0],action_arm_right[i])
        relative_action_arm_left_term = relative_pose(obs_arm_left[i][6:],action_arm_left[i])
        relative_action_arm_right_term = relative_pose(obs_arm_right[i][6:],action_arm_right[i])

        data_dict['/action/relative_arm_left'].append(relative_action_arm_left_term)
        data_dict['/action/relative_arm_right'].append(relative_action_arm_right_term)
        data_dict['/action/relative_hand_left'].append(action_hand_left[i]) ## 左手的不需要
        data_dict['/action/relative_hand_right'].append(action_hand_right[i]) ## 右手的不需要


        # 相机数据

        if args.use_depth_image:
            for cam_name in args.camera_names_depth:
                        ### 是否压缩图像 ######
                if args.use_compress_image:
                    data_dict[f'/observations/images_depth/{cam_name}'].append(compress_depth_image(image_dict_depth[cam_name][i]))
                else:
                    data_dict[f'/observations/images_depth/{cam_name}'].append(image_dict_depth[cam_name][i])
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in args.camera_names:
            if args.use_compress_image:
                data_dict[f'/observations/images/{cam_name}'].append(compress_image(image_dict[cam_name][i]))
            else:
                data_dict[f'/observations/images/{cam_name}'].append(image_dict[cam_name][i])


            ## 图像数据增广
            if args.use_augment:
                cv2.imwrite(f'/mnt1/zhangtianle/adam_rm_story/adam-rm-devel/act/output/image_{cam_name}_{episode_idx}.png', cv2.cvtColor(image_dict[cam_name][i], cv2.COLOR_RGB2BGR)) 
                ## 单纯数据增广，调对比度，色调之类的:
                image_temp_augment = image_dict[cam_name][i]
                image_temp_augment = torch.from_numpy(image_temp_augment)
                image_temp_augment = torch.einsum('h w c -> c h w', image_temp_augment)
                for transform in transformations_augment:
                    image_temp_augment = transform(image_temp_augment)
                image_temp_augment = torch.einsum('c h w -> h w c', image_temp_augment).numpy()
                cv2.imwrite(f'/mnt1/zhangtianle/adam_rm_story/adam-rm-devel/act/output/image_{cam_name}_augment_{episode_idx}.png', cv2.cvtColor(image_temp_augment, cv2.COLOR_RGB2BGR))
                data_dict[f'/observations/images/{cam_name}_augment'].append(image_temp_augment)


            
                ## 分割物体
                if cam_name == 'cam_body':
                    hqsam_act = hqsam_act_body
                    hqsam_act_robot =hqsam_act_body_robot
                elif cam_name == 'cam_head':
                    hqsam_act = hqsam_act_head
                    hqsam_act_robot =hqsam_act_head_robot
                
                image_temp = image_dict[cam_name][i]
                h,w = image_temp.shape[0],image_temp.shape[1]
                text_prompt = args.sam_text[0].split(',')



                if args.tracking_model == 'cutie':
                    image_mask = hqsam_act.track_segment_cutie_act(image_temp,text_prompt, i)    
                elif args.tracking_model == 'siammask':
                    image_mask = hqsam_act.track_segment_act(image_temp,text_prompt, i)      
                else:
                    image_mask = hqsam_act.segment_act(image_temp,text_prompt)
                image_mask = np.expand_dims(np.clip(image_mask, 0, 1),axis=2)



                ### 如果第一针没有，则每次识别是否有机器手，如果有的话，就加上去
                # if robot_tracking_step > 0 :
                #     image_mask_robot_hand = hqsam_act_robot.track_segment_cutie_act(image_temp,['robot hand'], robot_tracking_step)
                #     robot_tracking_step +=1
                # else:
                #     image_mask_robot_hand = image_mask_robot_hand = hqsam_act_robot.segment_act(image_temp,['robot hand'])
                #     if image_mask_robot_hand is not None:
                #         image_mask_robot_hand = hqsam_act_robot.track_segment_cutie_act(image_temp,['robot hand'], robot_tracking_step)
                #         robot_tracking_step +=1

                # if image_mask_robot_hand is None:
                #     pass
                # else:
                #     image_mask_robot_hand = np.expand_dims(np.clip(image_mask_robot_hand, 0, 1),axis=2)
                #     image_mask += image_mask_robot_hand


                # if image_mask.size == 0:
                #     image_mask = np.zeros((h,w))
                # elif args.tracking_model == 'cutie':
                #     image_mask = image_mask
                # else:
                #     image_mask = image_mask[0]
                #print(image_mask)


                
                image_temp_mask = image_temp * image_mask

                ## 变化背景
                # image_temp_other = image_temp * (1-image_mask)
                # image_temp_other = torch.from_numpy(image_temp_other)
                # image_temp_other = torch.einsum('h w c -> c h w', image_temp_other)
                # for transform in transformations_mask:
                #     image_temp_other = transform(image_temp_other)
                # image_temp_other = torch.einsum('c h w -> h w c', image_temp_other).numpy()

                # image_temp_all = image_temp_other.copy()

                # image_temp_all[image_mask[:, :, 0] > 0] = image_temp_mask[image_mask[:, :, 0] > 0]
                ### 分割后的图像增强

                #cv2.imwrite(f'/mnt1/zhangtianle/adam_rm_story/adam-rm-devel/act/output/image_{cam_name}_{episode_idx}.png', cv2.cvtColor(image_temp, cv2.COLOR_RGB2BGR)) 
                #cv2.imwrite(f'/mnt1/zhangtianle/adam_story/adam-devel/act/output/image_mask_{episode_idx}.png', cv2.cvtColor(image_temp_mask, cv2.COLOR_RGB2BGR))
                #cv2.imwrite(f'/mnt1/zhangtianle/adam_story/adam-devel/act/output/image_mask_others_{episode_idx}.png', cv2.cvtColor(image_temp_other, cv2.COLOR_RGB2BGR)) 
                cv2.imwrite(f'/mnt1/zhangtianle/adam_rm_story/adam-rm-devel/act/output/image_{cam_name}_mask_{episode_idx}.png', cv2.cvtColor(image_temp_mask, cv2.COLOR_RGB2BGR))
                #cv2.imwrite(f'/mnt1/zhangtianle/adam_rm_story/adam-rm-devel/act/output/image_{cam_name}_mask_{episode_idx}_{i}.png', cv2.cvtColor(image_temp_mask, cv2.COLOR_RGB2BGR))   # 保存图像
                
                data_dict[f'/observations/images/{cam_name}_mask'].append(image_temp_mask)



        if args.use_pointcloud:
            ## 点云
            data_dict['/observations/pointcloud'].append(point_cloud[i])



        # 解决点云维度不统一的问题
    if args.use_pointcloud:
        max_rows = max(array.shape[0] for array in data_dict['/observations/pointcloud'])
        max_cols = max(array.shape[1] for array in data_dict['/observations/pointcloud'])
        print('max rows: ', max_rows)
        print('max_cols: ', max_cols)
        padded_arrays = []
        for array in data_dict['/observations/pointcloud']:
            padded_array = np.zeros((max_rows, max_cols))
            padded_array[:array.shape[0], :array.shape[1]] = array
            padded_arrays.append(padded_array)
        data_dict['/observations/pointcloud'] = np.stack(padded_arrays, axis=0)
    
    t0 = time.time()


    if args.use_augment_for_save_video:

        absolute_path_for_subtask = save_dataset_dir+"/episode_v_"+str(episode_idx)
        if not os.path.exists(absolute_path_for_subtask):
            os.makedirs(absolute_path_for_subtask)

        for cam_name in args.camera_names:
            output_video_path = absolute_path_for_subtask +'/'+cam_name+'_'+str(episode_idx)+'.mp4'
            numpy_images_to_video(image_dict[cam_name],output_video_path, absolute_path = output_video_path)
            output_video_path_augment = absolute_path_for_subtask +'/'+cam_name+'_augment_'+str(episode_idx)+'.mp4'
            numpy_images_to_video(np.array(data_dict[f'/observations/images/{cam_name}_augment']), output_video_path, absolute_path = output_video_path_augment)
            output_video_path_mask = absolute_path_for_subtask +'/'+cam_name+'_mask_'+str(episode_idx)+'.mp4'
            numpy_images_to_video(np.array(data_dict[f'/observations/images/{cam_name}_mask']), output_video_path, absolute_path = output_video_path_mask)



    if args.use_save_data:
        absolute_path_for_subtask = save_dataset_dir+"/episode_v_"+str(episode_idx)
        if not os.path.exists(absolute_path_for_subtask):
            os.makedirs(absolute_path_for_subtask)
        ## 画出动作左臂:
        arm_position_plot(np.array(data_dict['/action/relative_arm_left']),'relateive_act_arm_left_'+str(episode_idx), absolute_path = absolute_path_for_subtask)
        arm_position_plot(np.array(data_dict['/observations/relative_arm_left']),'relateive_obs_arm_left_'+str(episode_idx), absolute_path = absolute_path_for_subtask)

        ## 画出动作右臂:
        arm_position_plot(np.array(data_dict['/action/relative_arm_right']),'relative_act_arm_right_'+str(episode_idx), absolute_path = absolute_path_for_subtask)
        arm_position_plot(np.array(data_dict['/observations/relative_arm_right']),'relative_obs_arm_right_'+str(episode_idx), absolute_path = absolute_path_for_subtask)
        
        
        # ## 画出左手
        # ## 画出观测左手
        hand_angle_plot(np.array(data_dict['/action/relative_hand_left'])[:,:6], 'relative_act_hand_left_'+str(episode_idx),absolute_path = absolute_path_for_subtask)
        hand_angle_plot(np.array(data_dict['/observations/relative_hand_left'])[:,:6], 'relative_obs_hand_left_'+str(episode_idx),absolute_path = absolute_path_for_subtask)

        ## 画观测力
        hand_angle_plot(np.array(data_dict['/observations/relative_hand_left'])[:,6:], 'relative_obs_hand_force_left_'+str(episode_idx),absolute_path = absolute_path_for_subtask)
        hand_angle_plot(np.array(data_dict['/observations/relative_hand_right'])[:,6:], 'relative_obs_hand_force_right_'+str(episode_idx),absolute_path = absolute_path_for_subtask)

        # ## 画出观测右手
        hand_angle_plot(np.array(data_dict['/action/relative_hand_right'])[:,:6], 'relative_act_hand_right_'+str(episode_idx),absolute_path = absolute_path_for_subtask)
        hand_angle_plot(np.array(data_dict['/observations/relative_hand_right'])[:,:6], 'relative_obs_hand_right_'+str(episode_idx),absolute_path = absolute_path_for_subtask)
        
        
        for cam_name in args.camera_names:
            output_video_path = absolute_path_for_subtask +'/'+cam_name+'_'+str(episode_idx)+'.mp4'
            if args.use_compress_image:
                numpy_images_to_video(decompress_images(image_dict[cam_name]),absolute_path_for_subtask+'/'+cam_name+'_', absolute_path = output_video_path)
            else:
                numpy_images_to_video(image_dict[cam_name],absolute_path_for_subtask+'/'+cam_name+'_', absolute_path = output_video_path)
        
    
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        #
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        act = root.create_group('action')
        image = obs.create_group('images')
        for cam_name in args.camera_names:

            if args.use_compress_image:
                max_bytes = max(len(img) for img in data_dict[f'/observations/images/{cam_name}'])

                _  = image.create_dataset(
                    cam_name,
                    (data_size,),  # 形状为 (数据数量, 最大字节数)
                    dtype=h5py.string_dtype(length=max_bytes), 
                    compression="gzip",
                    compression_opts=1
                )
            else:
                _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3),)


            if args.use_augment:
                _ = image.create_dataset(cam_name +"_augment", (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
                _ = image.create_dataset(cam_name +"_mask", (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names_depth:
                # _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint8', ## uint8
                #                              chunks=(1, 480, 640), )
                if args.use_compress_image:
                    max_bytes = max(len(img) for img in data_dict[f'/observations/images_depth/{cam_name}'])
                    
                    _  = image_depth.create_dataset(
                        cam_name,
                        (data_size,),  # 形状为 (数据数量, 最大字节数)
                        dtype=h5py.string_dtype(length=max_bytes), 
                        compression="gzip",
                        compression_opts=1
                    )
                else:
                    _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint8', chunks=(1, 480, 640),)

        if args.use_pointcloud:
            _ = obs.create_dataset('pointcloud', (data_size, max_rows, max_cols), dtype='float32', chunks=(1, max_rows, max_cols),)

        _ = obs.create_dataset('arm_left', (data_size, 13))
        _ = obs.create_dataset('arm_right', (data_size, 13))
        _ = obs.create_dataset('hand_left', (data_size, 18))
        _ = obs.create_dataset('hand_right', (data_size, 18))
        _ = obs.create_dataset('relative_arm_left', (data_size, 6))
        _ = obs.create_dataset('relative_arm_right', (data_size, 6))
        _ = obs.create_dataset('relative_hand_left', (data_size, 12))
        _ = obs.create_dataset('relative_hand_right', (data_size, 12))



        _ = obs.create_dataset('instruction', (data_size),dtype='S100')
        _ = act.create_dataset('arm_left', (data_size, 7))
        _ = act.create_dataset('arm_right', (data_size, 7))
        _ = act.create_dataset('hand_left', (data_size, 6))
        _ = act.create_dataset('hand_right', (data_size, 6))
        _ = act.create_dataset('relative_arm_left', (data_size, 6))
        _ = act.create_dataset('relative_arm_right', (data_size, 6))
        _ = act.create_dataset('relative_hand_left', (data_size, 6))
        _ = act.create_dataset('relative_hand_right', (data_size, 6))
        _ = root.create_dataset('base_action', (data_size, 2))
        _ = root.create_dataset('reward', (1))

        # data_dict写入h5py.File
        for name, array in data_dict.items():   # 名字+值
            print(name)
            print(np.shape(array))
            root[name][...] = array

    print(f'Saving: {time.time() - t0:.1f} secs', dataset_path)

def smooth_hand_data(hand_data, window_size=5, sigma=2.0):
    """对手部数据进行平滑处理
    
    Args:
        hand_data (np.ndarray): 手部数据，形状为 (timesteps, features)
        window_size (int): 滑动窗口大小，必须是奇数
        sigma (float): 高斯滤波的标准差
        
    Returns:
        np.ndarray: 平滑后的手部数据
    """
    if not isinstance(hand_data, np.ndarray):
        hand_data = np.array(hand_data)
    
    # 确保窗口大小是奇数
    if window_size % 2 == 0:
        window_size += 1
        
    # 分离角度数据和力数据
    angle_data = hand_data[:, :6]  # 前6维是角度数据
    force_data = hand_data[:, 12:] if hand_data.shape[1] > 12 else None  # 后6维是力数据
    
    # 创建高斯核
    x = np.linspace(-3, 3, window_size)
    gaussian_kernel = np.exp(-x**2 / (2*sigma**2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    
    # 对角度数据进行平滑处理
    smoothed_angle = np.zeros_like(angle_data)
    for i in range(angle_data.shape[1]):
        # 使用卷积进行平滑
        padded_data = np.pad(angle_data[:, i], (window_size//2, window_size//2), mode='edge')
        smoothed_angle[:, i] = np.convolve(padded_data, gaussian_kernel, mode='valid')
    
    # 如果有力数据，也进行平滑处理
    if force_data is not None:
        smoothed_force = np.zeros_like(force_data)
        for i in range(force_data.shape[1]):
            # 中值滤波去除异常值
            force_median = signal.medfilt(force_data[:, i], kernel_size=3)
            # 高斯平滑
            padded_data = np.pad(force_median, (window_size//2, window_size//2), mode='edge')
            smoothed_force[:, i] = np.convolve(padded_data, gaussian_kernel, mode='valid')
            
        # 合并其他数据（如果有的话，比如6-12维的数据）
        other_data = hand_data[:, 6:12]
        # 组合所有数据
        smoothed_data = np.concatenate([smoothed_angle, other_data, smoothed_force], axis=1)
    else:
        # 合并其他数据（如果有的话，比如6-12维的数据）
        other_data = hand_data[:, 6:12]
        smoothed_data = np.concatenate([smoothed_angle, other_data], axis=1)
        
    return smoothed_data

def main(args):
    # import ipdb;ipdb.set_trace()

    # args.camera_names = eval(args.camera_names)

    if args.use_augment:
        from models.grounded_segment_anything.EfficientSAM.grounded_light_hqsam_function import hqsam

        gdc_path = os.path.dirname(os.path.abspath(__file__)) + "/grounded_segment_anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        gd_model_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +'/models/groundingdino/groundingdino_swint_ogc.pth'
        hqsam_model_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +'/models/sam_hq/sam_hq_vit_tiny.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from models.Cutie.cutie.inference.inference_core import InferenceCore
        from models.Cutie.cutie.utils.get_default_model import get_default_model
        # obtain the Cutie model with default parameters -- skipping hydra configuration
        cutie = get_default_model()
        # Typically, use one InferenceCore per video
        processor = InferenceCore(cutie, cfg=cutie.cfg)

        siammask = None
        siammask_hp = None
        hqsam_act_body = hqsam(gdc_path, gd_model_path, hqsam_model_path, device, siammask, siammask_hp, processor, debug=args.debug)
        hqsam_act_head = hqsam(gdc_path, gd_model_path, hqsam_model_path, device, siammask, siammask_hp, processor, debug=args.debug)
        hqsam_act_body_robot = hqsam(gdc_path, gd_model_path, hqsam_model_path, device, siammask, siammask_hp, processor, debug=args.debug)
        hqsam_act_head_robot = hqsam(gdc_path, gd_model_path, hqsam_model_path, device, siammask, siammask_hp, processor, debug=args.debug)


        ## 图像增广参数设置
        ratio = 0.98 # 0.95
        original_size = np.array([480,640])
        transformations_augment = [
            #transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
            #transforms.Resize(original_size.tolist(), antialias=True),
            #transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False), # -5 5
            transforms.ColorJitter(brightness=(0.2, 1.2), contrast=0.4, saturation=0.5), #, hue=0.08)
            #transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.2)
            ]
        
        # transformations_mask = [
        #     RandomWhiteBlackColorJitter(
        #     white_color_range=((200, 200, 200), (255, 255, 255)),  # 白色区域的随机颜色范围
        #     black_color_range=((0, 0, 0), (50, 50, 50))            # 黑色区域的随机颜色范围
        #     ),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        #     transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.5, 2.5)),
        #     AdjustOpacity(opacity=0.5)  # 调整透明度
        #     #transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.2)
        # ]
    else:
        transformations_augment = None
        hqsam_act_body = None
        hqsam_act_head = None
        hqsam_act_body_robot = None
        hqsam_act_head_robot = None

    
    
    episode_len = args.episode_len
    load_dataset_dir = args.load_dataset_dir
    load_task_name = args.load_task_name


    save_dataset_dir = args.save_dataset_dir
    save_task_name = args.save_task_name
    if not os.path.exists(save_dataset_dir):
        os.makedirs(save_dataset_dir)
    
    load_dataset_path = os.path.join(load_dataset_dir, load_task_name)

    ## 处理哪几个episode id
    ## 1. 自定义
    episode_list = [38]
    
    ## 2. 读取
    all_hdf5_list = find_all_hdf5(load_dataset_path,False)

    #print(all_hdf5_list)

    camera_names_default = copy.deepcopy(args.camera_names)
    
    list_episode_idx = []
    #for episode_idx in tqdm(range(episode_len)):
    #for episode_idx in tqdm(episode_list):
    for hdf5_one in tqdm(all_hdf5_list):

        
        hdf5_one_split = hdf5_one_split = hdf5_one.split('.hdf5')[0].split('episode_')
        episode_idx = hdf5_one_split[1]
        
        if int(episode_idx) in list_episode_idx:
            continue
        else:
            list_episode_idx.append(int(episode_idx))
            print(list_episode_idx)
        
        dataset_name = f'episode_{episode_idx}'
        obs_arm_left, obs_arm_right, obs_hand_left, obs_hand_right, image_dict, image_dict_depth, point_cloud, \
        state, action_arm_left, action_arm_right, action_hand_left, action_hand_right, base_action, reward = load_hdf5(args, load_dataset_path, dataset_name)
        


        
        
        # if reward[0] == 0:
        #     continue
        
        
        image_dict_new = {}
        if args.use_split_head_image:
            
            args.camera_names = camera_names_default
            for camera in args.camera_names:
                if camera == 'cam_head':
                    decompress_image = decompress_images(image_dict[camera])
                    # 原始图像大小为 (366, 720, 2560, 3)
                    # 将图像从中间割开
                    batch_size, height, width, channels = decompress_image.shape  # 这里的形状是 (366, 720, 2560, 3)
                    mid_width = width // 2
                    left_half = decompress_image[:, :, :mid_width, :]  # 左半部分
                    right_half = decompress_image[:, :, mid_width:, :]  # 右半部分
                    #print(np.shape(right_half))
                    #target_size = (640,360)
                    # 调整大小，并压缩
                    left_half_images = []
                    for img in left_half:
                        #resized_img = cv2.resize(img, target_size)
                        #print(np.shape(resized_img))
                        compressed_img = compress_image(img)
                        #print(compressed_img)
                        left_half_images.append(compressed_img)
                        
                    right_half_images = []
                    for img in right_half:
                        #resized_img = cv2.resize(img, target_size)
                        compressed_img = compress_image(img)
                        right_half_images.append(compressed_img)
                    image_dict_new['cam_head_left'] = np.array(left_half_images)
                    image_dict_new['cam_head_right'] = np.array(right_half_images)
                else:
                    image_dict_new[camera] = image_dict[camera]
            image_dict = image_dict_new

            args.camera_names = ['cam_body','cam_head_left','cam_head_right']
        
        
        print(image_dict.keys())



        if args.use_save_data:
            
            save_path_for_all = load_dataset_path + '/plot_data' 
            if not os.path.exists(save_path_for_all):
                os.makedirs(save_path_for_all)
            ## 存储图像为视频:
            for camera in args.camera_names:
                output_video_path =save_path_for_all +'/'+camera+'_'+str(episode_idx)+'.mp4'
                if args.use_compress_image:
                    #print(np.shape(image_dict[camera]))
                    aaa = decompress_images(image_dict[camera])
                    #print(np.shape(aaa))
                    numpy_images_to_video(aaa,output_video_path, absolute_path = output_video_path)
                else:
                    numpy_images_to_video(image_dict[camera],output_video_path, absolute_path = output_video_path)
            
            # 关闭存储数据
            # if args.use_depth_image:
            #     for camera in args.camera_names_depth:
            #         output_depth_path = save_path_for_all +'/'+camera+'_depth_'+str(episode_idx)+'.avi'
            #         if args.use_compress_image:
            #             save_depth_maps_as_video(decompress_depth_images(image_dict_depth[camera]),save_path_for_all+'/'+cam_name+'_', absolute_path = output_depth_path)
            #         else:
            #             save_depth_maps_as_video(image_dict_depth[camera],save_path_for_all+'/'+cam_name+'_', absolute_path = output_depth_path)

            ## 画出观测左臂:
            arm_position_plot(obs_arm_left[:,6:9],'obs_arm_left_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画出观测右臂:
            arm_position_plot(obs_arm_right[:,6:9],'obs_arm_right_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画出观测左手
            hand_angle_plot(obs_hand_left[:,:6], 'obs_hand_left_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画出观测右手
            hand_angle_plot(obs_hand_right[:,:6], 'obs_hand_right_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画观测力
            hand_angle_plot(obs_hand_left[:,12:], 'obs_hand_force_left_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画观测力
            hand_angle_plot(obs_hand_right[:,12:], 'obs_hand_force_right_'+str(episode_idx), absolute_path = save_path_for_all) 

            ## 画出动作左臂:
            arm_position_plot(action_arm_left[:,:3],'act_arm_left_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画出动作右臂:
            arm_position_plot(action_arm_right[:,:3],'act_arm_right_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画出动作左手
            hand_angle_plot(action_hand_left[:,:], 'act_hand_left_'+str(episode_idx), absolute_path = save_path_for_all)

            ## 画出动作右手
            hand_angle_plot(action_hand_right[:,:], 'act_hand_right_'+str(episode_idx), absolute_path = save_path_for_all)
            
            
            ## 存储state
            save_byte_string_array_to_file(state,"state_"+str(episode_idx), absolute_path = save_path_for_all)
            
            #continue
            ## 画出点云
            #render_time_series_point_clouds_to_video(point_cloud[:,:,:3], output_file='/output_dataprocess/'+'_point_cloud_'+str(i)+'.avi')
            

        print('hdf5 loaded!!', episode_idx)
        #hand teleoperation ongoing





        
        index_list = [0]
        
        if args.use_split:
            for j in range(len(state)-1):
                j+=1
                if 'hand teleoperation' in state[j].decode('utf-8') and 'hand teleoperation' not in state[j-1].decode('utf-8'): # 头
                    index_list.append(j)
                elif 'hand teleoperation' not in state[j].decode('utf-8') and 'hand teleoperation' in state[j-1].decode('utf-8'): # 尾
                    index_list.append(j)
        
        index_list.append(len(state))
        
        
        for k in range(len(index_list)-1):
            
            initial_index = index_list[k]
            final_index = index_list[k+1]
            
            obs_arm_left_term = obs_arm_left[initial_index:final_index]
            obs_arm_right_term = obs_arm_right[initial_index:final_index]
            obs_hand_left_term = obs_hand_left[initial_index:final_index]
            obs_hand_right_term = obs_hand_right[initial_index:final_index]
            image_dict_term = {}
            image_dict_depth_term = {}
            
            for cam_name in args.camera_names:
                image_dict_term[cam_name] = image_dict[cam_name][initial_index:final_index]
            if args.use_depth_image:
                for cam_name in args.camera_names_depth:
                    image_dict_depth_term[cam_name] = image_dict_depth[cam_name][initial_index:final_index]
            if args.use_pointcloud:
                point_cloud_term = point_cloud[initial_index:final_index]
            else:
                point_cloud_term = None
            state_term = state[initial_index:final_index]
            action_arm_left_term = action_arm_left[initial_index:final_index]
            action_arm_right_term = action_arm_right[initial_index:final_index]
            action_hand_left_term = action_hand_left[initial_index:final_index]
            action_hand_right_term = action_hand_right[initial_index:final_index]
            base_action_term = base_action[initial_index:final_index]
            reward_term = reward[0]

            ##分子任务
            task_name_index = k // 2
            save_task_name_term = save_task_name[task_name_index]
            save_dataset_dir_1 = os.path.join(save_dataset_dir, save_task_name_term)

            ##分规划和操作
            #import ipdb;ipdb.set_trace();
            if args.use_split:
                if k % 2 ==0:
                    phase_name = 'plan'
                else:
                    phase_name = 'teach'
                    
                save_dataset_dir_2 = os.path.join(save_dataset_dir_1, phase_name)
                

                save_dataset_dir_3 = os.path.join(save_dataset_dir_2, "episodes_" + phase_name)

                if not os.path.exists(save_dataset_dir_3):
                    os.makedirs(save_dataset_dir_3)
                save_dataset_path = os.path.join(save_dataset_dir_3, "episode_" + str(episode_idx))
            else:
                if not os.path.exists(save_dataset_dir_1):
                    os.makedirs(save_dataset_dir_1)
                save_dataset_dir_2 = save_dataset_dir_1
                phase_name = ''
                save_dataset_path = os.path.join(save_dataset_dir_1, "episode_" + str(episode_idx))




            
    
        

            save_data(args, save_dataset_path, obs_arm_left_term, obs_arm_right_term, obs_hand_left_term, obs_hand_right_term, image_dict_term, image_dict_depth_term, point_cloud_term, 
                    state_term, action_arm_left_term, action_arm_right_term, action_hand_left_term, action_hand_right_term, base_action_term, reward_term, save_dataset_dir_2, episode_idx,
                    phase_name, hqsam_act_body, hqsam_act_head, hqsam_act_body_robot, hqsam_act_head_robot, transformations_augment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',default=True, required=False)
    parser.add_argument('--use_pointcloud', action='store', type=bool, help='use_pointcloud',default=False, required=False)
    

    parser.add_argument('--episode_len', action='store', type=int, help='episode length', default=25, required=False)
    parser.add_argument('--debug', action='store', type=bool, help='debug', default=False, required=False)


    parser.add_argument('--camera_names', action='store', type=str, help='camera_names', default=['cam_body','cam_head'], required=False) # ['cam_body', 'cam_head', 'cam_head2']
    parser.add_argument('--camera_names_depth', action='store', type=str, help='camera_names_depth', default=['cam_body','cam_head'], required=False) # ['cam_body', 'cam_head']


    #### 数据路径 for 204 ####
    # parser.add_argument('--use_save_data', action='store', type=bool, help='use_save_data',default=True, required=False)
    # parser.add_argument('--load_dataset_dir', default='/mnt1/zhangtianle/adam_rm_story/datasets/take_cup/', type=str, required=False, help='load dataset directory')
    # parser.add_argument('--load_task_name', default='episodes_teach', type=str, required=False, help='load dataset directory') # take_and_place_cup Colorful Rubik's Cube orange_platter iphone_charge rubik_cube_play


    # parser.add_argument('--save_dataset_dir', default='/mnt1/zhangtianle/adam_rm_story/datasets/take_cup_new/', type=str, required=False, help='load dataset directory')
    # parser.add_argument('--save_task_name', default=['episodes_teach5'], type=str, required=False, help='load dataset directory')
    # parser.add_argument('--use_split', action='store', type=bool, help='use_split', default=False, required=False)

    ### 数据路径 for A6000 ####
    parser.add_argument('--use_save_data', action='store', type=bool, help='use_save_data',default=False, required=False)
    parser.add_argument('--load_dataset_dir', default='/mnt/hil-serl/datasets/data_take_mango', type=str, required=False, help='load dataset directory') # data_screen_zed
    parser.add_argument('--load_task_name', default='aloha_mobile_dummy', type=str, required=False, help='load dataset directory') # take_and_place_cup Colorful Rubik's Cube orange_platter iphone_charge rubik_cube_play

    parser.add_argument('--save_dataset_dir', default='/mnt/hil-serl/datasets/data_take_mango/', type=str, required=False, help='load dataset directory')
    parser.add_argument('--save_task_name', default=['pre_process_debug'], type=str, required=False, help='load dataset directory') #['take_cup','place_cup']
    parser.add_argument('--use_split', action='store', type=bool, help='use_split', default=False, required=False)
    #['press_screen_coffee_zed','press_screen_confirm_zed','press_screen_placed_zed','press_screen_return_zed']
    #['take_cup_zed','place_cup_zed']
    
    
    #### 数据增强 ####
    parser.add_argument('--tracking_model', action='store', type=str, help='tracking_model', default='cutie', required=False) # cutie siammask
    parser.add_argument('--sam_text', action='store', type=str, nargs='+', default=['white cup'], help='a', required=False) # apple,orange   mobile phone   Colorful Rubik's Cube
    parser.add_argument('--use_augment', action='store', type=bool, help='use_augment', default=False, required=False)
    parser.add_argument('--use_augment_for_save_video', action='store', type=bool, help='use_augment_for_save_video', default=False, required=False)

    #### 是否压缩图像
    parser.add_argument('--use_compress_image', action='store', type=bool, help='use_compress_image', default=True, required=False)

    #### 是否处理手力数据
    parser.add_argument('--use_hand_force_process', action='store', type=bool, help='use_hand_force_process', default=False, required=False)

    ### 是否将手的数据进行光滑处理
    parser.add_argument('--use_smooth_hand_data', action='store', type=bool, help='use_smooth_hand_data', default=False, required=False)
    
    
    ### 是否分割head图像
    parser.add_argument('--use_split_head_image', action='store', type=bool, help='use_split_head_image', default=True, required=False)
    main(parser.parse_args())



    ### 数据处理代码###
    # python data_process.py --
