import argparse
import copy
import os
import cv2
import open3d as o3d

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def bev_to_points(bev_seg, xbound, ybound, save_path):
    indices = np.argwhere(bev_seg.squeeze() == 1)
    x_indices, y_indices = indices[:, 0], indices[:, 1]
    x_coords = x_indices * xbound[2] + xbound[0]
    y_coords = y_indices * ybound[2] + ybound[0]
    z_coords = np.full(x_coords.shape, -0.11)

    points = np.vstack((x_coords, y_coords, z_coords)).T

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(points)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(
        save_path,
        o3d_points
    )

def image2video(image_folder1, image_folder2, video_path):
    # 读取第一个文件夹中的图片文件名
    images1 = [img for img in os.listdir(image_folder1) if img.endswith(".png")]
    images1.sort()

    # 读取第二个文件夹中的图片文件名
    images2 = [img for img in os.listdir(image_folder2) if img.endswith(".png")]
    images2.sort()

    # 确保两个列表中的图像数量和顺序相同
    assert len(images1) == len(images2), "Image lists are not the same!"

    # 获取一张图片来确定单个图像的分辨率
    frame1 = cv2.imread(os.path.join(image_folder1, images1[0]))
    height, width, layers = frame1.shape

    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, 10, (2 * width, height + 50))

    # 将两组图片逐一读取，拼接，然后写入视频
    for img1, img2 in zip(images1, images2):
        frame1 = cv2.imread(os.path.join(image_folder1, img1))
        frame2 = cv2.imread(os.path.join(image_folder2, img2))
        # 旋转图像180度
        frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
        frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
        # 水平拼接两个图像
        combined_frame = cv2.hconcat([frame1, frame2])
        # 创建新帧，增加额外空间
        combined_frame = cv2.copyMakeBorder(combined_frame, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))

        # 添加文本
        cv2.putText(combined_frame, 'masks_bev_gt', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(combined_frame, 'masks_bev_pred', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)
        # 写入视频
        video.write(combined_frame)

    # 释放资源
    video.release()

def main() -> None:
    os.environ['MASTER_HOST'] = "localhost" + ":" + "12356"
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--only-video", action="store_true")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    wrap_fp16_model(model)
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    xbound, ybound = cfg.test_pipeline[3].xbound, cfg.test_pipeline[3].ybound

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}".format(metas["timestamp"])

        with torch.inference_mode():
            outputs = model(**data)

        name_gt = name + "_gt"
        masks_gt = data["gt_masks_bev"].data[0].numpy()
        masks_gt = masks_gt.astype(np.bool_)

        name_pred = name + "_pred"
        masks_pred = outputs[0]["masks_bev"].numpy()

        # masks_pred = np.loadtxt("runs/seg_camera_only_resnet50_ge_bev_output_scope_0.5/head.map.classifier.output.cpp.txt").reshape(1, 200, 200)

        masks_pred = masks_pred >= args.map_score

        visualize_map(
            os.path.join(args.out_dir, "seg", "pred", f"{name_pred}.png"),
            masks_pred,
            classes=cfg.map_classes,
        )

        visualize_map(
            os.path.join(args.out_dir, "seg", "gt", f"{name_gt}.png"),
            masks_gt,
            classes=cfg.map_classes,
        )

        bev_to_points(
            masks_pred,
            xbound,
            ybound,
            os.path.join(args.out_dir, "seg", "pred_points", f"{name_pred}.pcd"),
        )

    # 两个图片文件夹路径
    image_folder1 = os.path.join(args.out_dir, "seg", "gt")
    image_folder2 = os.path.join(args.out_dir, "seg", "pred")
    # 视频输出路径
    video_path = os.path.join(args.out_dir, "seg", "masks_bev_diff.avi")
    image2video(image_folder1, image_folder2, video_path)


if __name__ == "__main__":
    main()
