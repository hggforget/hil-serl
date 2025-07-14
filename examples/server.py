# -- coding: UTF-8
# """
# #!/usr/bin/python3 #!/home/lin/software/miniconda3/envs/aloha/bin/python
# """

# import torch
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pickle
import argparse
from einops import rearrange
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("./")

# from utils.utils_default import compute_dict_mean, set_seed, detach_dict # helper functions
# from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy,HITPolicy
# from torchvision import transforms
import time
import threading
import math
import threading
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, JointState
# from utils.visualization import visualize_image
from collections import deque

import grpc
from concurrent import futures
import image_transfer_pb2
import image_transfer_pb2_grpc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from dual_arm_msgs.msg import Hand_Status, CartePos   ####  jd@jd-System-Product-Name:~/jd_kitchen/catkin_hand$ source devel/setup.bash
from sensor_msgs.msg import Image, JointState, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
# from models.status_detection.status_det import StatusDet
import random
import argparse

import sys
sys.path.append("./")
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# from infer_save_data import infer_save_data


def compress_image(img, quality=100):
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


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    
    Parameters:
    roll (float): Rotation around the x-axis in radians.
    pitch (float): Rotation around the y-axis in radians.
    yaw (float): Rotation around the z-axis in radians.
    
    Returns:
    tuple: Quaternion (w, x, y, z)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)




def extract_focus_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    coords = np.where(mask)
    if len(coords[0]) > 0:
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        rgb_region = image[y_min:y_max + 1, x_min - 1:x_max + 1]
        rgb_region = cv2.resize(rgb_region, (640, 360))
        return rgb_region
    else:
        return image


class RosOperator:
    def __init__(self):
        self.hand_focues_left_image = None
        self.hand_focues_right_image = None
        self.bridge = None
        self.init()
        self.init_ros()

    def init(self):
        self.hand_focues_left_image = None
        self.hand_focues_right_image = None
        self.bridge = CvBridge()
        
        self.left_arm_feedback_pose_msg = PoseStamped()
        self.right_arm_feedback_pose_msg = PoseStamped()
        self.left_hand_feebback_msg = Hand_Status()
        self.right_hand_feebback_msg = Hand_Status()
        self.left_arm_feedback_pose_msg.header = Header()
        self.right_arm_feedback_pose_msg.header = Header()
        self.left_hand_feebback_msg.header = Header()
        self.right_hand_feebback_msg.header = Header()
        
    def doDetect(self,left_arm_pos, left_hand_angle, right_arm_pos, right_hand_angle, image_head_left, image_head_right):
        cur_timestamp = rospy.Time.now()  # 设置时间戳
        self.left_arm_feedback_pose_msg.header.stamp = cur_timestamp 
        self.left_arm_feedback_pose_msg.pose.position.x = left_arm_pos[0]
        self.left_arm_feedback_pose_msg.pose.position.y = left_arm_pos[1]
        self.left_arm_feedback_pose_msg.pose.position.z = left_arm_pos[2]
        self.left_arm_feedback_pose_msg.pose.orientation.x = left_arm_pos[3]
        self.left_arm_feedback_pose_msg.pose.orientation.y = left_arm_pos[4]
        self.left_arm_feedback_pose_msg.pose.orientation.z = left_arm_pos[5]
        self.left_arm_feedback_pose_msg.pose.orientation.w = left_arm_pos[6]
        self.left_arm_feedback_pose_publisher.publish(self.left_arm_feedback_pose_msg)

        self.right_arm_feedback_pose_msg.header.stamp = cur_timestamp 
        self.right_arm_feedback_pose_msg.pose.position.x = right_arm_pos[0]
        self.right_arm_feedback_pose_msg.pose.position.y = right_arm_pos[1]
        self.right_arm_feedback_pose_msg.pose.position.z = right_arm_pos[2]
        self.right_arm_feedback_pose_msg.pose.orientation.x = right_arm_pos[3]
        self.right_arm_feedback_pose_msg.pose.orientation.y = right_arm_pos[4]
        self.right_arm_feedback_pose_msg.pose.orientation.z = right_arm_pos[5]
        self.right_arm_feedback_pose_msg.pose.orientation.w = right_arm_pos[6]
        self.right_arm_feedback_pose_publisher.publish(self.right_arm_feedback_pose_msg)

        img_head_left_msg = self.bridge.cv2_to_imgmsg(image_head_left, 'passthrough')
        img_head_left_msg.header.stamp = cur_timestamp
        self.img_head_left_publisher.publish(img_head_left_msg)
        img_head_right_msg = self.bridge.cv2_to_imgmsg(image_head_right, 'passthrough')
        img_head_right_msg.header.stamp = cur_timestamp
        self.img_head_right_publisher.publish(img_head_right_msg)

        self.left_hand_feebback_msg.header.stamp = cur_timestamp 
        self.right_hand_feebback_msg.header.stamp = cur_timestamp 
        for j in range(6):
            self.left_hand_feebback_msg.hand_angle[j] = np.uint32(left_hand_angle[j])
            self.right_hand_feebback_msg.hand_angle[j] = np.uint32(right_hand_angle[j])
        self.left_hand_feedback_publisher.publish(self.left_hand_feebback_msg)
        self.right_hand_feedback_publisher.publish(self.right_hand_feebback_msg)


    def get_frame(self):

        

        hand_focus_left_image = self.bridge.imgmsg_to_cv2(self.hand_focues_left_image, 'passthrough')
        hand_focus_right_image = self.bridge.imgmsg_to_cv2(self.hand_focues_right_image, 'passthrough')



        return (hand_focus_left_image, hand_focus_right_image)
    
    def image_callback_left(self, msg):
        
        self.hand_focues_left_image = msg
    
    def image_callback_right(self, msg):
        self.hand_focues_right_image = msg

    
    def init_ros(self):
        rospy.init_node('imitation_learning_subscriber', anonymous=True, disable_signals=True)
        rospy.Subscriber('/camera_head/left/hand_focus', Image, self.image_callback_left, queue_size=1)
        rospy.Subscriber('/camera_head/right/hand_focus', Image, self.image_callback_right, queue_size=1)
        
        
        self.img_head_left_publisher = rospy.Publisher('/camera_head/zed_node/left/image_rect_color', Image, queue_size=10)
        self.img_head_right_publisher = rospy.Publisher('/camera_head/zed_node/right/image_rect_color', Image, queue_size=10)
        self.left_hand_feedback_publisher = rospy.Publisher('/l_arm/rm_driver/Udp_Hand_Status', Hand_Status, queue_size=10)
        self.right_hand_feedback_publisher = rospy.Publisher('/r_arm/rm_driver/Udp_Hand_Status', Hand_Status, queue_size=10)
        self.left_arm_feedback_pose_publisher = rospy.Publisher("/l_arm/rm_driver/Pose_State", PoseStamped, queue_size=10)
        self.right_arm_feedback_pose_publisher = rospy.Publisher("/r_arm/rm_driver/Pose_State", PoseStamped, queue_size=10)


class ImageTransferServicer(image_transfer_pb2_grpc.ImageTransferServicer):
    def __init__(self):
        
        
        self.ros_operator = RosOperator()
        self.rate = rospy.Rate(120)
        
        self.args = get_arguments()
        task_config = {
            'camera_names': ['cam_body','cam_head_left'] # 'camera_names': ['cam_body', 'cam_head_left', 'cam_head_right']
            }
        self.camera_names = task_config['camera_names']

        ### 初始的状态
        self.start_left_arm_pose = None
        self.start_right_arm_pose = None
        self.start_left_hand_force = None
        self.start_right_hand_force = None


    def UploadImageAndData(self, request, context):

        time_start = time.time()
        # 需要Instruction
        instruction = request.instruction_data
        
        # 需要一个时间戳
        t = request.step_data
        
        
        image_body = decompress_image(request.image_body_data)
        #image_head = decompress_image(request.image_head_data)
        
        ### 将image_head 进行分离
        # height, width, channels = image_head.shape  # 这里的形状是 (720, 2560, 3)
        # mid_width = width // 2
        # image_head_left = image_head[:, :mid_width, :]  # 左半部分
        # image_head_right = image_head[:, mid_width:, :]  # 右半部分
        
        
        
        # Print the received float data
        # left_arm = request.left_float_data
        # left_arm_joints = left_arm[:6]
        # left_arm_pos = left_arm[6:13]
        # left_hand_angle = left_arm[13:19]
        # left_hand_pos = left_arm[19:25]
        # left_hand_force = left_arm[25:31]


        right_arm = request.right_arm_data
        right_arm_joints = right_arm[:6]
        right_arm_pos = right_arm[6:13] ## 用这个
        right_hand_angle = right_arm[13:19] ## 用这个
        right_hand_pos = right_arm[19:25]
        right_hand_force = right_arm[25:31]
        
        print(right_arm)

        # if self.args.use_hand_focus_image or self.args.use_status_detection:
        #     ### 先发送给一航
        #     self.ros_operator.doDetect(left_arm_pos, left_hand_angle, right_arm_pos, right_hand_angle, image_head_left, image_head_right)
        

        # target_size = (640,360)
        # image_head_left = cv2.resize(image_head_left, target_size)
        # image_head_right = cv2.resize(image_head_right, target_size)
        
        
        # if self.args.use_hand_focus_image or self.args.use_status_detection:
        #     while self.ros_operator.hand_focues_left_image is None or self.ros_operator.hand_focues_right_image is None:
        #         self.rate.sleep()
                
        #     (hand_focus_left_image, hand_focus_right_image) = self.ros_operator.get_frame()
            
            
        #     if self.args.use_focus_image_resize:
        #         hand_focus_left_image = extract_focus_region(hand_focus_left_image)
        #         hand_focus_right_image = extract_focus_region(hand_focus_right_image)
                
        #     hand_focus_left_image = cv2.resize(hand_focus_left_image, target_size)
        #     hand_focus_right_image = cv2.resize(hand_focus_right_image, target_size)
        #     self.ros_operator.hand_focues_left_image = None
        #     self.ros_operator.hand_focues_right_image = None
            
        
        
        if t == 0:
            #self.start_left_arm_pose = left_arm_pos
            #self.start_left_hand_force = left_hand_force

            self.start_right_arm_pose = right_arm_pos
            self.start_right_hand_force = right_hand_force

        # obs_dict = {}
        # obs_dict['cam_body'] = np.array(image_body)
        # obs_dict['cam_head_left'] = np.array(image_head_left)
        # obs_dict['cam_head_right'] = np.array(image_head_right)
        
        
        # if self.args.use_hand_focus_image:
        #     obs_dict['hand_focus_left'] = np.array(hand_focus_left_image)
        #     obs_dict['hand_focus_right'] = np.array(hand_focus_right_image)
            
        # obs_dict['left_arm_pos'] = np.array(left_arm_pos)
        # obs_dict['left_hand_angle'] = np.array(left_hand_angle)
        # obs_dict['left_hand_force'] = np.array(left_hand_force)

        # obs_dict['right_arm_pos'] = np.array(right_arm_pos)
        # obs_dict['right_hand_angle'] = np.array(right_hand_angle)
        # obs_dict['right_hand_force'] = np.array(right_hand_force)

        # obs_dict['start_left_arm_pose'] = np.array(self.start_left_arm_pose)
        # obs_dict['start_left_hand_force'] = np.array(self.start_left_hand_force)

        # obs_dict['start_right_arm_pose'] = np.array(self.start_right_arm_pose)
        # obs_dict['start_right_hand_force'] = np.array(self.start_right_hand_force)

        ######################进行推理#####################


        # print("action: ", action)
        # print("next_action: ", next_action)
        ## 提取动作
        # if self.args.use_one_arm == 'left_arm':
        #     left_action = action[:12]
        #     right_action = np.zeros(12)
        # elif self.args.use_one_arm == 'right_arm':
        #     left_action = np.zeros(12)
        #     right_action = action[:12]
        # elif self.args.use_one_arm == 'both':
        #     left_action = action[:12]
        #     right_action = action[12:]
        
        left_action = np.zeros(12)
        right_action = np.zeros(12)
        # ####左臂和左手
        relative_left_arm = left_action[:6]
        left_hand = left_action[6:]
        ####右臂和右手
        relative_right_arm = right_action[:6]
        right_hand = right_action[6:]
        
        # #相对变绝对
        if self.args.use_one_arm == 'left_arm':
            left_arm = absolute_pose(self.start_left_arm_pose,relative_left_arm)
            right_arm = np.zeros(7)
        elif self.args.use_one_arm == 'right_arm':  
            left_arm = np.zeros(7)
            right_arm = absolute_pose(self.start_right_arm_pose,relative_right_arm)
        else:
            left_arm = absolute_pose(self.start_left_arm_pose,relative_left_arm)
            right_arm = absolute_pose(self.start_right_arm_pose,relative_right_arm)
        
        

        
        final_action = np.concatenate([left_arm, left_hand, right_arm, right_hand])
        status_det_label = 1
       
        print("time: ", time.time() - time_start)
                
        return image_transfer_pb2.ImageDataResponse(float_data=final_action, status_det_label=status_det_label)
    


def get_arguments():
    parser = argparse.ArgumentParser()
    
    #### 只使用一直臂
    parser.add_argument('--use_one_arm', action='store', type=str, help='use_one_arm', default='right_arm', required=False) # left, right, both
    
    ## 是否使用手部聚焦图像
    parser.add_argument('--use_hand_focus_image', action='store', type=bool, help='use_hand_focus_image', default=False, required=False)
    
    ## 是否使用聚焦图像resize
    parser.add_argument('--use_focus_image_resize', action='store', type=bool, help='use_focus_image_resize', default=False, required=False)
    
    args = parser.parse_args()
    return args
    

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_transfer_pb2_grpc.add_ImageTransferServicer_to_server(ImageTransferServicer(), server)
    server.add_insecure_port('192.168.10.50:50055')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':

    serve()