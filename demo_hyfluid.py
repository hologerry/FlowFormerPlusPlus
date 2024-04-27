import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cv2 import imwrite
from PIL import Image
from tqdm.contrib import tzip

from configs.submissions import get_cfg as get_submission_cfg
from core.FlowFormer import build_flowformer
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.frame_utils import writeFlow
from core.utils.utils import InputPadder
from evaluate_FlowFormer_tile import compute_grid_indices, compute_weight


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()


def viz(img, flo, out_imfile):
    time_in = time.time()
    # img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    time_convert = time.time()
    # print("flow to image", time_convert - time_in)
    out_filename = out_imfile.replace(".jpg", ".png")
    imwrite(out_filename, flo)
    time_save = time.time()
    # print("time saving image", time_save - time_convert)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.savefig(out_imfile)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]]/255.0)
    # cv2.waitKey()


def save_flow(flo, out_flowfile):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    writeFlow(out_flowfile, flo)


@torch.no_grad()
def demo(args):

    # model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))
    cfg = get_submission_cfg()
    print(cfg)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model = model.module
    model.cuda()
    model.eval()

    os.makedirs(args.fw_output_path, exist_ok=True)
    os.makedirs(args.bw_output_path, exist_ok=True)

    images = glob.glob(os.path.join(args.path, "*.png")) + glob.glob(os.path.join(args.path, "*.jpg"))

    images = sorted(images)
    print(f"find {len(images)} images")

    IMAGE_SIZE = [576, 1024]
    TRAIN_SIZE = [432, 960]

    sigma = 0.05

    if args.stage == "fw":
        for imfile1, imfile2 in tzip(images[:-1], images[1:]):
            time_s = time.time()
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            time_data = time.time()
            # print("data loading", time_data - time_s)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            time_data_processing = time.time()
            # print("data processing", time_data_processing - time_data)

            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

            flows = 0
            flow_count = 0

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
                flow_pre, flow_low = model(image1_tile, image2_tile)

                padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow = flows / flow_count

            print("flow shape", flow.shape)
            # fw_flow_low, fw_flow_up = model(image1, image2)
            # torch.cuda.synchronize()
            # time_flow = time.time()
            # # print("flow prediction", time_flow - time_data_processing, "iters", args.iters)

            fw_out_imfile1 = imfile1.replace(args.path, args.fw_output_path)
            viz(image1, flow, fw_out_imfile1)
            # time_save = time.time()
            # print("saving", time_save - time_flow)

    elif args.stage == "bw":
        for imfile_p, imfile_c in tzip(images[:-1], images[1:]):
            image_p = load_image(imfile_p)
            image_c = load_image(imfile_c)

            padder = InputPadder(image_p.shape)
            image_p, image_c = padder.pad(image_p, image_c)

            bw_flow_low, bw_flow_up = model(image_c, image_p)

            bw_out_imfile1 = imfile_c.replace(args.path, args.bw_output_path)
            viz(image_c, bw_flow_up, bw_out_imfile1)

    elif args.stage == "fwt":
        for imfile_p, imfile_c in tzip(images[:-1], images[1:]):
            fwt_out_flowfile1 = imfile_c.replace("img", "fwt_flow").replace(".jpg", ".flo")

            image_p = load_image(imfile_p)
            image_c = load_image(imfile_c)

            padder = InputPadder(image_p.shape)
            image_p, image_c = padder.pad(image_p, image_c)

            fwt_flow_low, fwt_flow_up = model(image_p, image_c)

            save_flow(fwt_flow_up, fwt_out_flowfile1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint", default="models/raft-kitti.pth")
    parser.add_argument(
        "--path",
        help="dataset for evaluation",
        default="/home/yuegao/Dynamics/dynamics_processing/vis_hyfluid/hyfluid_teaser_frames_stable_crop",
    )
    parser.add_argument("--stage", help="forward or backward", default="fw")
    parser.add_argument("--fw_output_path", help="output path for evaluation")
    parser.add_argument("--bw_output_path", help="output path for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--iters", type=int, help="iterations for raft", default=20)
    parser.add_argument("--alternate_corr", action="store_true", help="use efficient correlation implementation")
    args = parser.parse_args()
    args.fw_output_path = "/home/yuegao/Dynamics/dynamics_processing/vis_hyfluid/hyfluid_teaser_flow_former_kitti_fw"
    args.bw_output_path = "/home/yuegao/Dynamics/dynamics_processing/vis_hyfluid/hyfluid_teaser_flow_former_kitti_bw"

    demo(args)
