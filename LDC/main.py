import argparse
import os
import shutil
from pathlib import Path

import cv2
from cv2 import imread
import numpy as np

from utils import save_ldc_map
from ldc import LDC

def parse_args():
    parser = argparse.ArgumentParser('Undistortion Tool')
    parser.add_argument('--imp', type=str,
                        default=None, 
                        help='path of distorted images')
    parser.add_argument('--project', type=str,
                        required=True)
    parser.add_argument('--scene', type=str,
                        required=True,
                        help='front/front_stereo/side/fisheye')
    parser.add_argument('--paras_calib_p', type=str, 
                        default='paras_calib.xml')
    parser.add_argument('--uparas_p', type=str,
                        default='uparas.json')
    parser.add_argument('--no_ldc', action='store_true')
    parser.add_argument('--dst', type=str, 
                        default='savep')
    args = parser.parse_args()
    
    if args.imp is None: args.imp = Path('data/images') / args.project / args.scene
    args.paras_calib_p = Path('data/paras') / args.project / args.scene / args.paras_calib_p
    args.uparas_p = Path('data/paras') / args.project / args.scene / args.uparas_p
    
    shutil.rmtree(args.dst)
    args.dst = Path(args.dst) / args.project / args.scene
    
    print(args)
    return args


def chery_fisheye_test():
    args = parse_args()
    ldc = LDC(paras_calib_p=args.paras_calib_p, 
            uparas_p=args.uparas_p, 
            project=args.project, 
            scene=args.scene,
            no_ldc=args.no_ldc)
    ldc.init_ldc_map()
    ldc.transform_ldc_map()
    save_ldc_map(ldc.scene2map, args.dst)
    # undist._load_extrinsic_paras(args.intrinsic)

    cnt = 0
    for rt, _, fnames in os.walk(args.imp / 'front'):
        if len(fnames) == 0: continue

        rt = Path(rt)
        for fname in fnames:
            if Path(fname).suffix not in ['.jpg', '.png', '.bmp']: continue
            cnt += 1
            imfp = rt / fname
            cam = rt.stem
            print('==> cnt: {} imgfp: {}'.format(cnt, imfp))

            # front rear left right
            im_front = cv2.imread(str(imfp))
            im_rear = cv2.imread(str(imfp).replace('front', 'rear').replace('vc10', 'vc12'))
            im_left = cv2.imread(str(imfp).replace('front', 'left').replace('vc10', 'vc9'))
            im_right = cv2.imread(str(imfp).replace('front', 'right').replace('vc10', 'vc11'))
            im = np.vstack([np.vstack([im_front, im_rear]), np.vstack([im_left, im_right])])

            im_ldc = ldc.remap(im, 'fisheye', mode='scene')

            imp_dst = args.dst / 'images' / rt.relative_to(args.imp)
            imfp_dst = imp_dst / Path(fname).with_suffix('.jpg')
            if not imp_dst.exists():
                imp_dst.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(imfp_dst.as_posix(), im_ldc)

if __name__ == '__main__':
    chery_fisheye_test()
            
