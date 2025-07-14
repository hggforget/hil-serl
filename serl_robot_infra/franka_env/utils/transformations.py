from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T

def construct_adjoint_matrix_from_euler(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix_from_euler(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T



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