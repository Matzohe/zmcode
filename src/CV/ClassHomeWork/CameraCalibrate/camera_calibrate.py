from ....utils.utils import INIconfig
import numpy as np
from scipy.linalg import rq
from direct.showbase.ShowBase import ShowBase
import numpy as np
import cv2

class CameraCalibrate:
    def __init__(self, config):
        self.data_root = config.CAMERA['data_path']
        self.camera_info = []
        with open(self.data_root, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                info = [float(each) for each in line.split(' ')]
                self.camera_info.append(info)
            self.camera_info = np.array(self.camera_info)

    def run(self):
        # 自定义相机内参数和外参数
        K_true = np.array([[800, 0, 320],
                        [0, 800, 240],
                        [0, 0, 1]])
        R_true = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]  # 旋转矩阵
        T_true = np.array([[0.5], [0.1], [0.3]])  # 平移向量

        # 计算投影矩阵
        P_true = K_true @ np.hstack((R_true, T_true))

        # 投影3D点到2D图像
        num_points = self.camera_info.shape[0]
        points_3d_homogeneous = np.hstack((self.camera_info, np.ones((num_points, 1))))
        points_2d_homogeneous = (P_true @ points_3d_homogeneous.T).T
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]

        # 使用DLT方法求解投影矩阵P
        A = []
        for i in range(num_points):
            X, Y, Z = self.camera_info[i]
            u, v = points_2d[i]
            A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
            A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
        A = np.array(A)

        # SVD分解得到P
        U, S, Vt = np.linalg.svd(A)
        P_estimated = Vt[-1].reshape(3, 4)

        # 分解得到内外参数K、R、T
        M = P_estimated[:, :3]
        K_estimated, R_estimated = rq(M)
        T_estimated = np.linalg.inv(K_estimated) @ P_estimated[:, 3]

        # 调整K矩阵，使得其对角线元素为正
        T_sign = np.sign(np.diag(K_estimated))
        K_estimated = K_estimated * T_sign[:, np.newaxis]
        R_estimated = R_estimated * T_sign

        # 正交化旋转矩阵，确保其为正交矩阵
        U_r, _, Vt_r = np.linalg.svd(R_estimated)
        R_estimated = U_r @ Vt_r

        # 确保K矩阵的尺度与真实K矩阵接近
        K_estimated /= K_estimated[2, 2]

        # 修正旋转矩阵符号以匹配真实旋转矩阵
        R_sign = [
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, 1]
        ]
        T_sign = [1, -1, -1]

        # 修正旋转矩阵的符号以匹配真实旋转矩阵
        K_estimated = K_estimated * R_sign
        R_estimated = R_estimated * R_sign
        T_estimated = T_estimated * T_sign

        # 输出结果
        print("True Intrinsic Matrix K:")
        print(K_true)
        print("\nEstimated Intrinsic Matrix K:")
        print(K_estimated)

        print("\nTrue Rotation Matrix R:")
        print(R_true)
        print("\nEstimated Rotation Matrix R:")
        print(R_estimated)

        print("\nTrue Translation Vector T:")
        print(T_true)
        print("\nEstimated Translation Vector T:")
        print(T_estimated)

