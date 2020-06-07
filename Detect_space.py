from __future__ import absolute_import
import argparse
import cv2
from detect_space_module import * 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='single')
    parser.add_argument('--image_path', default= '')
    parser.add_argument('--folder_path', default='')
    parser.add_argument('--config_path', default='')
    args = parser.parse_args()
    cfg = Config(args.config_path)
    
    if args.mode not in ['single','multiple']:
        sys.exit()
    elif args.mode == 'single':
        image = cv2.imread(args.image_path)
        process_in_single_cfg(image, cfg, True)
    else:
        for name in listdir(args.folder_path):
            path = osp.join(args.folder_path, name)
            image = cv2.imread(path)
            process_in_single_cfg(image, cfg, True)