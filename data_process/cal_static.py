import h5py
import os
import numpy as np

def calculate_statistics_from_hdf5(directory_path):
    # 存储所有值的列表
    all_values = []
    
    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.hdf5') or filename.endswith('.h5'):
            file_path = os.path.join(directory_path, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    # 检查key是否存在
                    key_string = '/observations/relative_hand_right'
                    if key_string in f:
                        # 读取数据并转换为numpy数组
                        data = f[key_string][:]
                        for row in data:
                            all_values.append(row)
                    else:
                        print(f"警告: 文件 {filename} 中没有找到 'relative_action_arm' 键")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    if all_values:
        # 转换为numpy数组以便计算统计值
        all_values = np.array(all_values)
        print(all_values.shape)
        
        # 计算统计值
        max_value = np.max(all_values,axis=0)
        min_value = np.min(all_values,axis=0)
        mean_value = np.mean(all_values,axis=0)
        std_value = np.std(all_values,axis=0)
        
        # 打印结果
        print("\n统计结果:")
        print(f"最大值: {max_value}")
        print(f"最小值: {min_value}")
        print(f"平均值: {mean_value}")
        print(f"标准差: {std_value}")
    else:
        print("没有找到任何数据")

if __name__ == "__main__":
    # 指定要处理的目录路径
    directory_path = "/mnt/hil-serl/datasets/data_take_cup_new_rl/pre_process_init"  # 请替换为实际的目录路径
    calculate_statistics_from_hdf5(directory_path)
