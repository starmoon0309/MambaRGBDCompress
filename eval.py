import argparse
import json
import math
import sys
import os
import time
import struct
import os.path as osp

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
import compressai

from compressai.zoo.pretrained import load_pretrained
from compressai.ops import compute_padding
from models.entropy_model import *
from torch.hub import load_state_dict_from_url
import re

from models.MambaRGBX_v5 import MambaRGBD
import random,cv2
from lib.utils import get_output_folder, AverageMeter, save_checkpoint, MyTestDataset, MyDataset, NYUDataset, NYUTestDataset
from torch.utils.data import Dataset, DataLoader

seed = 2026
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

save_path = ""



def collect_images(rootpath, catagory):
    path = Path(rootpath)  
    path = path / (catagory + "_test") 
    left_image_list = [f for f in (path / 'left').iterdir() if f.is_file()]
    right_image_list = [f for f in (path / 'right').iterdir() if f.is_file()]

    return [left_image_list, right_image_list]


def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)
    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg
    

def save_pic(data, h, w, path):
    if osp.exists(path):
        os.system("rm " + path)
        print("rm " + path)
    img = data[:h, :w, :]
    cv2.imwrite(path, img)



def compute_metrics_for_frame(
    org_frame: Tensor,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,):

    org_frame = (org_frame * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_frame - rec_frame).pow(2).mean()
    
    
    psnr_float = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)
    ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=max_val)
    return psnr_float, ms_ssim_float


def compute_bpp(likelihoods, num_pixels):
    bpp = sum(
        (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
        for likelihood in likelihoods.values()
    )
    return bpp


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def pad_gray_image(image_gray, pad):
    x = torch.from_numpy(image_gray).unsqueeze(0)
    x_padded = F.pad(x, pad, mode='constant', value=0)
    padded_image = x_padded.squeeze(0).byte().numpy()
    return padded_image



def read_gray_image_pad(filepath, pad):

    assert os.path.isfile(filepath)
    img1_ori = cv2.imread(filepath)
    img1_gray = cv2.cvtColor(img1_ori, cv2.COLOR_BGR2GRAY)
    padded_image_gray = pad_gray_image(img1_gray, pad)
    return padded_image_gray
    
    
def read_rgb_image(filepath):

    assert os.path.isfile(filepath)
    img1_ori = cv2.imread(filepath)
    img1 = cv2.cvtColor(img1_ori, cv2.COLOR_BGR2RGB)
    return img1
    
    
def image_transforms(img):
    return transforms.ToTensor()(img)

@torch.no_grad()
def eval_model(test_dataloader, IFrameCompressor, **args: Any) -> Dict[str, Any]:
    device = 'cuda'
    max_val = 2**8 - 1
    results = defaultdict(list)

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            rgb = batch[2].to(device)  
            depth = batch[0].to(device)  
            name_rgb = ''.join(batch[3]) 
            name_depth = ''.join(batch[1])
            num_pixels = rgb.size(2) * rgb.size(3)
            h, w = rgb.size(2), rgb.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)
            x_left_padded = F.pad(depth, pad, mode="constant", value=0)
            x_right_padded = F.pad(rgb, pad, mode="constant", value=0)
            out_net = IFrameCompressor(x_left_padded,x_right_padded)
            start = time.time()
            ddd = IFrameCompressor.compress(x_left_padded, x_right_padded)
            enc_time = time.time() - start
            start = time.time()
            out_dec = IFrameCompressor.decompress(ddd["strings"], ddd["shape"])
            dec_time = time.time() - start
            x_left_rec, x_right_rec = F.pad(out_dec["x_hat"][0], unpad), F.pad(out_dec["x_hat"][1], unpad)
            print("bpp0:",sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net['likelihoods1'].values()))   
            print("bpp1:",sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net['likelihoods2'].values())     )
            metrics = {}
            metrics["depth-psnr-float"], metrics["depth-ms-ssim-float"] = compute_metrics_for_frame(
                depth, x_left_rec, device, max_val)
            metrics["rgb-psnr-float"], metrics["rgb-ms-ssim-float"] = compute_metrics_for_frame(
                rgb, x_right_rec, device, max_val)
            metrics["psnr-float"] = (metrics["depth-psnr-float"]+metrics["rgb-psnr-float"])/2
            metrics["ms-ssim-float"] = (metrics["depth-ms-ssim-float"]+metrics["rgb-ms-ssim-float"])/2
            metrics["depth_bpp"] = torch.tensor(sum(len(s) for s in ddd["strings"][0]) * 8.0 / num_pixels) + torch.tensor(sum(len(s) for s in ddd["strings"][1]) * 8.0 / num_pixels)
            metrics["rgb_bpp"] = torch.tensor(sum(len(s) for s in ddd["strings"][2]) * 8.0 / num_pixels) + torch.tensor(sum(len(s) for s in ddd["strings"][3]) * 8.0 / num_pixels)
            metrics["bpp"] = (metrics["depth_bpp"] + metrics["rgb_bpp"])/2
            enc_time = torch.tensor(enc_time)
            dec_time = torch.tensor(dec_time)
            metrics["enc_time"] = enc_time
            metrics["dec_time"] = dec_time
            
            

            rec_d1 = torch.clamp(x_left_rec, min=0, max=1.0)
            rec_d1 = rec_d1.data[0].cpu().detach().numpy()
            rec_d1 = rec_d1.transpose(1, 2, 0) * 255.0
            rec_d1 = rec_d1.astype('uint8')

            rec_d2 = torch.clamp(x_right_rec, min=0, max=1.0)
            rec_d2 = rec_d2.data[0].cpu().detach().numpy()
            rec_d2 = rec_d2.transpose(1, 2, 0) * 255.0
            rec_d2 = rec_d2.astype('uint8')

            save_pic(rec_d1, h, w, save_path + "Depth/" + name_depth)
            save_pic(rec_d2, h, w, save_path + "RGB/" + name_rgb)

            for k, v in metrics.items():
                results[k].append(v)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


def run_inference(
    test_dataloader,
    IFrameCompressor: nn.Module, 
    outputdir: Path,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any):

    with amp.autocast(enabled=args["half"]):
        with torch.no_grad():
              if entropy_estimation:
                  metrics = eval_model_entropy_estimation(test_dataloader , IFrameCompressor, **args)
              else:
                  metrics = eval_model(test_dataloader , IFrameCompressor, **args)
    return metrics
    

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RGB-D image compression network evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="sequences directory")
    parser.add_argument("--output", type=str, help="output directory")
    parser.add_argument(
        "-im",
        "--IFrameModel",
        default="...",
        help="Model architecture (default: %(default)s)",
    )


    parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--half", action="store_true", help="use AMP")
    
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=16,
        help="Dataloaders threads (default: %(default)s)",
    )
    
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    device = "cuda"
    test_dataset = NYUTestDataset(input_rgb_path=args.dataset + 'sunrgbd_rgb/', input_depth_path=args.dataset + 'sunrgbd_depth/')
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=False)

    IFrameCompressor = MambaRGBD(192,320).cuda()
    IFrameCompressor = IFrameCompressor.to(device)
    if args.i_model_path:
        print("Loading model:", args.i_model_path)
        checkpoint = torch.load(args.i_model_path, map_location=device)
        IFrameCompressor.load_state_dict(checkpoint["state_dict"])
        
    IFrameCompressor.update(force=True)
    IFrameCompressor.eval()
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    args_dict = vars(args)
    

    trained_net = f"{args.metric}-{description}"
    metrics = run_inference(test_dataloader, IFrameCompressor, outputdir, trained_net=trained_net, description=description, **args_dict)
    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": f"{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{args.metric}-{description}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
