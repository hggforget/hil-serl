from concurrent import futures
import time
from typing import Optional
import grpc
import image_transfer_pb2
import image_transfer_pb2_grpc
from datetime import datetime

class RobotControlClient:
    """Robot Control gRPC客户端类"""
    
    def __init__(self):
        """
        初始化gRPC客户端
        
        Args:
            server_address: gRPC服务器地址，默认为localhost:50051
        """
        channel = grpc.insecure_channel('192.168.10.53:50056')
        self.stub = image_transfer_pb2_grpc.ImageTransferStub(channel)
        print('连接成功')
        
    def reset(self):
        request = image_transfer_pb2.ImageDataRequest(flag=0)
        response = self.stub.UploadImageAndData(request)
        
        return {
                "image_body": response.image_body,
                "image_head_left": response.image_head_left,
                "image_head_right": response.image_head_right,
                "right_arm_pose": response.right_arm_pose,
                "right_hand_angle": response.right_hand_angle,
                "left_arm_pose": response.left_arm_pose,
                "left_hand_angle": response.left_hand_angle,
            }
    
    def step(self,right_action=None,left_action=None):
        
        
        if right_action is not None:
            request = image_transfer_pb2.ImageDataRequest(flag=1, right_action=right_action)
        response = self.stub.UploadImageAndData(request)
        
        return {
                "image_body": response.image_body,
                "image_head_left": response.image_head_left,
                "image_head_right": response.image_head_right,
                "right_arm_pose": response.right_arm_pose,
                "right_hand_angle": response.right_hand_angle,
                "left_arm_pose": response.left_arm_pose,
                "left_hand_angle": response.left_hand_angle,
            }



if __name__ == "__main__":
    
    robot_client = RobotControlClient()
    while True:
        start_time = time.time()
        obs = robot_client.step(right_action=[0,0,0,0,0,0,0,0,0,0,0,0,0])
        #print(obs)
        print(time.time()-start_time)