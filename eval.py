"""
Testing scripts.

Authors: Hongjie Fang.
"""
import os
import yaml
import torch
import logging
import warnings
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.functions import to_device
from time import perf_counter
from datasets.transcg_v2 import TransCG
from torch.utils.data import Subset, DataLoader


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--cfg', '-c',
    default = os.path.join('configs', 'default.yaml'),
    help = 'path to the configuration file',
    type = str
)
args = parser.parse_args();
cfg_filename = args.cfg

with open(cfg_filename, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)

logger.info('Building dataloaders ...')
dataset = TransCG(data_dir='/opt/data/private/trans_cg/transcg',
    split = 'test',
    image_size=(320, 240) # (640, 320) , (320, 240)
)
dataset = Subset(dataset, indices=range(128, ) )
test_dataloader = DataLoader(dataset, batch_size=1, num_workers=8,)
# prediction_dir = "/opt/data/private/Marigold/output/eval/trancg/marigold_v4_mask_iter_016000/prediction"
prediction_dir = "/opt/data/private/Marigold/output/eval/trancg/marigold_v3_iter_011350"

metrics = builder.get_metrics()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test():
    logger.info('Start testing process.')
    # model.eval()
    metrics.clear()
    running_time = []
    with tqdm(test_dataloader) as pbar:
        for data_dict in pbar:
            # data_dict = to_device(data_dict, device)
            data_dict["depth_gt"] = data_dict["depth_gt"].to(device)
            data_dict["depth_gt_mask"] = data_dict["depth_gt_mask"].to(device)
            data_dict["zero_mask"] = data_dict["zero_mask"].to(device)
            with torch.no_grad():
                time_start = perf_counter()
                # res = model(data_dict['rgb'], data_dict['depth'])
                rgb_name = data_dict["rgb_relative_path"][0]
                # Load predictions
                rgb_basename = os.path.basename(rgb_name)
                view_id = rgb_name.split("/")[-2]
                scene_id = rgb_name.split("/")[-3]
                pred_basename = rgb_basename.replace("rgb", "depth").replace(".png", "-pred.npy")
                pred_path = os.path.join(prediction_dir, scene_id, view_id, pred_basename)
                depth_pred = np.load(pred_path)
                time_end = perf_counter()
                res = torch.from_numpy(depth_pred)[None]
                depth_scale = data_dict['depth_max'] - data_dict['depth_min']
                res = res * depth_scale.reshape(-1, 1, 1) + data_dict['depth_min'].reshape(-1, 1, 1)
                data_dict['pred'] = res.to(device)
                _ = metrics.evaluate_batch(data_dict, record = True)
            duration = time_end - time_start
            pbar.set_description('Time: {:.4f}s'.format(duration))
            running_time.append(duration)
    avg_running_time = np.stack(running_time).mean()
    logger.info('Finish testing process, average running time: {:.4f}s'.format(avg_running_time))
    metrics_result = metrics.get_results()
    metrics.display_results()
    return metrics_result


if __name__ == '__main__':
    test()
