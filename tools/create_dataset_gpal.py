import os
import yaml
import numpy as np
import pickle
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    pkl_path = "/home/junzhiguan/bevfusion/data/se_seg"
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    data_root = "/data0/Ground_Elements_Datasets/data/2023-05-31/2023-05-31-11-04-46"

    ge_dir = os.path.join(data_root, 'annotation', 'ground_element')
    ge_files = sorted(os.listdir(ge_dir), key=lambda x: float(os.path.splitext(x)[0]))
    TIMESTAMP_LIST = [ge_file.split(".pcd")[0] for ge_file in ge_files]

    dataset = {"infos": [], "metadata": {"version": "v1.0-mini"}}
    for timestamp in tqdm(TIMESTAMP_LIST):
        data = {}
        ge_path = os.path.join(ge_dir, timestamp+".pcd")
        data["timestamp"] = float(timestamp)
        data["ge_path"] = ge_path

        # for CBGSDataset samples duplication
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        data["gt_names"] = np.array(CLASSES)
        data["valid_flag"] = np.array([True, True, True, True, True, True, True, True, True, True])
        # 以上代码为了适配nuscenes的格式， 需要gt_name 和 valid_flag 这样的参数，但是在seg map里面用不到
        data["cams"] = {}
        cams_type = ["front_left", "front_right", "left", "right", "back_left", "back_right"]
        for cam_type in cams_type:
            camera_path = os.path.join(data_root, "data", f"img_{cam_type}", f"{timestamp}.jpg")
            type = f"CAM_{cam_type.upper()}"

            # 解析参数
            camera_paras_path = os.path.join(data_root, "parameters/id4cameraparameters", f"{cam_type}.yml")
            fs = cv2.FileStorage(camera_paras_path, cv2.FILE_STORAGE_READ)
            # 读取内参数据
            camera_matrix_node = fs.getNode("camera_matrix")
            camera_intrinsics = camera_matrix_node.mat()
            # 读取畸变系数
            distortion_coefficients_node = fs.getNode("distortion_coefficients")
            distortion_coefficients = distortion_coefficients_node.mat()

            # 用于训练的图像的路径
            undistort_image_dir = os.path.join(data_root, "data", "undistorted_image", f"img_{cam_type}")
            if not os.path.exists(undistort_image_dir):
                os.makedirs(undistort_image_dir)
            undistort_image_path = os.path.join(undistort_image_dir, f"{timestamp}.jpg")

            # 图像去畸变并保存到磁盘 （如果已经生成就不需要执行了，因为去畸变保存磁盘实在太慢了）
            image = cv2.imread(camera_path)
            undistorted_image = cv2.undistort(image, camera_intrinsics, distortion_coefficients)
            cv2.imwrite(undistort_image_path, undistorted_image)

            # 读取ego2cam translation 和 rotation
            t_vec = fs.getNode("t_vec")
            ego2sensor_translation = t_vec.mat()
            r_mat = fs.getNode("r_mat")
            ego2sensor_rotation = r_mat.mat()

            # ego2cam transform matrix
            ego2sensor_T = np.zeros((4, 4))
            ego2sensor_T[:3, :3] = ego2sensor_rotation
            ego2sensor_T[:3, 3] = ego2sensor_translation.flatten()
            ego2sensor_T[3, 3] = 1

            # 计算 cam2ego transform matrix
            sensor2ego_T = np.linalg.inv(ego2sensor_T)
            # 计算 cam2lidar transform matrix, lidar点云和ego同坐标系
            sensor2lidar_T = sensor2ego_T.copy()

            data["cams"][type] = {
                        "data_path": undistort_image_path,
                        "type": type,
                        "camera_intrinsics": camera_intrinsics,
                        "sensor2ego_T": sensor2ego_T,
                        "sensor2lidar_T": sensor2lidar_T
                }

        dataset["infos"].append(data)

    with open(os.path.join(pkl_path, "nuscenes_infos_train.pkl"), "wb") as file:
        pickle.dump(dataset, file)

    # with open(os.path.join(pkl_path, "nuscenes_infos_train.pkl"), "rb") as file:
    #     dataset = pickle.load(file)

    # print(dataset)






