import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/data0/Ground_Elements_Datasets/data")
    parser.add_argument("--pkl-path", type=str, default="/home/junzhiguan/bevfusion/data/se_seg")
    parser.add_argument("--mode", type=str, default="train", help="train, val, infer")
    parser.add_argument("--sensor-layout", type=str, default="7v", help="6v, 7v")
    parser.add_argument("--undistort", action="store_true", help="")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.mode == "train":
        date_list = ["2024-02-05"]
        bagname_list = [[
                        "2024-02-05-14-34-19", "2024-02-05-14-44-19", "2024-02-05-14-54-19", "2024-02-05-15-04-19", "2024-02-05-15-14-19", \
                        "2024-02-05-15-26-30", "2024-02-05-15-36-30",  "2024-02-05-15-56-31", "2024-02-05-16-06-31", "2024-02-05-15-46-31"
                        ]
                        # ["2024-02-06-10-29-02", "2024-02-06-10-39-02", "2024-02-06-10-49-02", "2024-02-06-10-59-02", "2024-02-06-11-09-02", "2024-02-06-11-19-02"],
                        # ["2024-02-07-10-53-10"]
                        ]
    if args.mode == "val":
        date_list = ["2024-02-05"]
        bagname_list = [["2024-02-05-14-34-19", "2024-02-05-14-44-19", "2024-02-05-15-04-19"],
                        ]
    if args.mode == "infer":
        if args.sensor_layout == "6v":
            date_list = ["2023-05-31"]
            bagname_list = [["2023-05-31-11-04-46"]]
        if args.sensor_layout == "7v":
            date_list = ["2024-02-05"]
            bagname_list = [["2024-02-05-14-34-19"]]

    pkl_path = args.pkl_path
    os.makedirs(pkl_path, exist_ok=True)
    data_dir = args.data_dir

    if args.sensor_layout == "7v":
        cams_type = ["front_120", "back", "left_front", "left_back", "right_front", "right_back"]
    if args.sensor_layout == "6v":
        cams_type = ["front_left", "front_right", "left", "right", "back_left", "back_right"]

    dataset = {"infos": [], "metadata": {"version": "v1.0"}}
    for date_idx, date in enumerate(date_list):
        for bagname in bagname_list[date_idx]:
            data_root = os.path.join(data_dir, date, bagname)
            print(f"start process {data_root} ...")

            ge_dir = os.path.join(data_root, 'annotation', 'ground_element')
            ge_files = sorted(os.listdir(ge_dir), key=lambda x: float(os.path.splitext(x)[0]))
            TIMESTAMP_LIST_tmp = [ge_file.split(".pcd")[0] for ge_file in ge_files]

            if args.mode == "train" or args.mode == "val":
                TIMESTAMP_LIST = []
                if os.path.exists(os.path.join(data_root, 'annotation', 'sample.txt')):
                    print(f"open exists sample.txt in path: {os.path.join(data_root, 'annotation', 'sample.txt')}")
                    with open(os.path.join(data_root, 'annotation', 'sample.txt'), "r") as file:
                        lines = file.readlines()
                        for line in lines:
                            TIMESTAMP_LIST.append(line.strip('\n'))
                else:
                    print(f"random select 1200 samples in {os.path.join(data_root, 'annotation', 'sample.txt')}")
                    # 5倍降采样提取sample
                    indices = random.sample(range(len(TIMESTAMP_LIST_tmp)), 1200)
                    TIMESTAMP_LIST = [TIMESTAMP_LIST_tmp[idx] for idx in indices]
                    with open(os.path.join(data_root, 'annotation', 'sample.txt'), "w") as file:
                        for ts in TIMESTAMP_LIST:
                            file.write(ts + '\n')
            else:
                TIMESTAMP_LIST = TIMESTAMP_LIST_tmp

            for idx, timestamp in enumerate(tqdm(TIMESTAMP_LIST)):
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
                for cam_type in cams_type:
                    camera_path = os.path.join(data_root, "data", f"img_{cam_type}", f"{timestamp}.jpg")
                    type = f"CAM_{cam_type.upper()}"

                    # 解析参数
                    if args.sensor_layout == "7v":
                        camera_paras_path = os.path.join(data_dir, "calibration-master@fa4bb20d1b0/ID4/camera/2024_02_04", f"{cam_type}.yaml")
                    if args.sensor_layout == "6v":
                        camera_paras_path = os.path.join(data_dir, "id4cameraparameters", f"{cam_type}.yml")
                    fs = cv2.FileStorage(camera_paras_path, cv2.FILE_STORAGE_READ)
                    # 读取内参数据
                    camera_matrix_node = fs.getNode("camera_matrix")
                    camera_intrinsics = camera_matrix_node.mat()
                    # 读取畸变系数
                    distortion_coefficients_node = fs.getNode("distortion_coefficients")
                    distortion_coefficients = distortion_coefficients_node.mat()

                    if args.undistort:
                        # 用于训练的图像的路径
                        undistort_image_dir = os.path.join(data_root, "data", "undistorted_image", f"img_{cam_type}")
                        os.makedirs(undistort_image_dir, exist_ok=True)
                        if len(os.listdir(undistort_image_dir)) == 0:
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

                    if args.undistort:
                        data["cams"][type] = {
                                    "data_path": undistort_image_path,
                                    "type": type,
                                    "camera_intrinsics": camera_intrinsics,
                                    "sensor2ego_T": sensor2ego_T,
                                    "sensor2lidar_T": sensor2lidar_T
                            }
                    else:
                        data["cams"][type] = {
                            "data_path": camera_path,
                            "type": type,
                            "camera_intrinsics": camera_intrinsics,
                            "sensor2ego_T": sensor2ego_T,
                            "sensor2lidar_T": sensor2lidar_T
                        }

                dataset["infos"].append(data)
    if args.mode == "train":
        if args.undistort:
            with open(os.path.join(pkl_path, f"ge_infos_train_{args.sensor_layout}_{cams_type[0]}_undist.pkl"), "wb") as file:
                pickle.dump(dataset, file)
        else:
            with open(os.path.join(pkl_path, f"ge_infos_train_{args.sensor_layout}_{cams_type[0]}.pkl"), "wb") as file:
                pickle.dump(dataset, file)
    if args.mode == "val":
        if args.undistort:
            with open(os.path.join(pkl_path, f"ge_infos_val_{args.sensor_layout}_{cams_type[0]}_undist.pkl"), "wb") as file:
                pickle.dump(dataset, file)
        else:
            with open(os.path.join(pkl_path, f"ge_infos_val_{args.sensor_layout}_{cams_type[0]}.pkl"), "wb") as file:
                pickle.dump(dataset, file)
    if args.mode == "infer":
        if args.undistort:
            with open(os.path.join(pkl_path, f"ge_infos_infer_{args.sensor_layout}_{cams_type[0]}_undist.pkl"), "wb") as file:
                pickle.dump(dataset, file)
        else:
            with open(os.path.join(pkl_path, f"ge_infos_infer_{args.sensor_layout}_{cams_type[0]}.pkl"), "wb") as file:
                pickle.dump(dataset, file)





