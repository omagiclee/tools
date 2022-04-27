import cv2
import numpy as np

from utils import load_intrinsic_paras, load_json


class LDC:
    def __init__(self, paras_calib_p, uparas_p, project, scene, no_ldc=False):
        self.paras_calib_p = paras_calib_p
        self.project = project
        self.scene = scene
        self.imsize_calib = None
        self.no_ldc = no_ldc

        self.intrinsics = {}
        self.cam2uparas = {}

        self.cam2uparas = load_json(uparas_p)
        if self.no_ldc:
            self.intrinsics = {cam: None for cam in self.cam2uparas}
        else:   
            self.intrinsics = load_intrinsic_paras(self.paras_calib_p, self.project)

        # saic: left_front, left_rear, right_front, right_rear
        # chery: 
        #   front_stereo: fov120, fov30
        #   fisheye: left, right, front, rear
        # self.extrinsics = load_extrinsic_paras(paras_calib_p, project)
        self.cam2map  = {}
        self.scaled_cam2map  = {}
        self.scene2map = {}
        self._init_imsize_calib()

    def _init_imsize_calib(self):
        if self.project == 'saic':
            if self.scene == 'side':
                self.imsize_calib = [1920, 1280]
        if self.project == 'chery':
            if self.scene == 'front_stereo':
                self.imsize_calib = [1920, 1280]   
            if self.scene == 'fisheye':
                self.imsize_calib = [1280, 800]
        if self.project == 'mona':
            pass

    def remap(self, im, key, mode=None):
        if mode == 'scale':
            mapx, mapy = self.scaled_cam2map[key]
        elif mode == 'scene':
            mapx, mapy = self.scene2map[key]
        else:
            mapx, mapy = self.cam2map[key]
        dst = cv2.remap(im, mapx, mapy, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
        return dst

    # def undistort(self, img, cam):
    #     imh, imw = img.shape[:-1]
    #     assert(imw==self.imsize[0] and imh==self.imsize[1])
    #     mtx, dist = self.intrinsics[cam]
    #     # imsize = img.shape[:2][::-1]
    #     # mtx_new, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imsize, alpha, imsize)
    #     mtx_new = mtx.copy()
    #     dst = cv2.undistort(img, mtx, dist, None, mtx_new)

    def init_ldc_map(self):
        for cam in self.intrinsics:
            if self.no_ldc:
                x = np.arange(0, self.imsize_calib[0])
                y = np.arange(0, self.imsize_calib[1])
                mapx, mapy = np.meshgrid(x, y)
                mapx = mapx.astype(np.float32)
                mapy = mapy.astype(np.float32)
            else:
                mtx, dist = self.intrinsics[cam]
                imsize_new = self.imsize_calib
                
                cam_uparas = self.cam2uparas[cam]
                # mtx_new, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imsize_new, 0, self.imsize, 1)
                # Generate mtx_new
                fx_new = cam_uparas['mtx_new']['fx']
                fy_new = cam_uparas['mtx_new']['fy']
                cx_new = cam_uparas['mtx_new']['cx']
                cy_new = cam_uparas['mtx_new']['cy']
                mtx_new = np.array([[fx_new, 0, cx_new],
                                    [0, fy_new, cy_new],
                                    [0, 0, 1]])
                mtx_new[0, 2] += cam_uparas['cxcy_offset']['cx']
                mtx_new[1, 2] += cam_uparas['cxcy_offset']['cy']

                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx_new, imsize_new, cv2.CV_32FC1)
            self.cam2map[cam] = (mapx, mapy)

    def transform_ldc_map(self):
        top, bottom, left, right, w_tgt, h_tgt = 0, 0, 0, 0, 0, 0
        if self.project == 'chery':
            if self.scene == 'front_stereo':
                top, bottom = 90, 38
                left, right = 0, 0
                w_tgt, h_tgt = 0, 0
            if self.scene == 'fisheye':
                top, bottom = 0, 0
                left, right = 0, 0
                w_tgt, h_tgt = 448, 280
        if self.project == 'mona':
            if self.scene == 'front_stereo':
                top, bottom = 90, 38
                left, right = 0, 0
                w_tgt, h_tgt = 0, 0
            if self.scene == 'side':
                top, bottom = 128, 0
                left, right = 0, 0
                w_tgt = 640
                h_tgt = 384
        if self.project == 'saic':
            if self.scene == 'side':
                pass

        # Crop -> Scale
        for cam in self.cam2map:
            mapx, mapy = self.cam2map[cam]
            h_src, w_src = mapx.shape
            mapx = mapx[top:h_src-bottom, left:w_src-right]
            mapy = mapy[top:h_src-bottom, left:w_src-right]
            mapy -= top

            h_src_crop, w_src_crop = mapx.shape
            scale_h = h_tgt / h_src_crop
            scale_w = w_tgt / w_src_crop
            print('cam: {}, src.shape: {}, src_crop.shape: {}, scale: {}, tgt.shape: {}'
                    .format(cam, (w_tgt, h_tgt), (w_src, h_src), (scale_w, scale_h), (w_src_crop, h_src_crop)))
            scaled_mapx = cv2.resize(mapx, (w_tgt, h_tgt))
            scaled_mapy = cv2.resize(mapy, (w_tgt, h_tgt))
            self.scaled_cam2map[cam] = (scaled_mapx, scaled_mapy)
  
        # Stack
        if self.project == 'saic':
            if self.scene == 'side':
                mapx_downsample = cv2.resize(mapx[200:-120, :], (512, 256))
                mapy_downsample = cv2.resize(mapy[200:-120, :], (512, 256))
                if cam in ['left_front', 'right_rear']:
                    left = 896
                if cam in ['left_rear', 'right_front']:
                    left = 0
                mapx_roi = cv2.resize(mapx[496:496+288, left:left+1024], (512, 144))
                mapy_roi = cv2.resize(mapy[496:496+288, left:left+1024], (512, 144))
                mapx_out = np.vstack([mapx_roi, mapx_downsample])
                mapy_out = np.vstack([mapy_roi, mapy_downsample])

        if self.project == 'chery':
            if self.scene == 'front_stereo':
                # front_stereo: fov120, fov30
                mapx_fov120, mapy_fov120 = self.scaled_cam2map['fov120']
                mapx_fov30, mapy_fov30 = self.scaled_cam2map['fov30']
                mapx_out = np.vstack([mapx_fov120, mapx_fov30])
                mapy_out = np.vstack([mapy_fov120, mapy_fov30])

            if self.scene == 'fisheye':
                # fisheye: front rear left right
                mapx_left, mapy_left = self.scaled_cam2map['left']
                mapx_right, mapy_right = self.scaled_cam2map['right']
                mapx_front, mapy_front = self.scaled_cam2map['front']
                mapx_rear, mapy_rear = self.scaled_cam2map['rear']
                mapy_rear += h_src
                mapy_left += 2*h_src
                mapy_right += 3*h_src
                
                mapx_out = np.vstack([mapx_front, mapx_rear, mapx_left, mapx_right])
                mapy_out = np.vstack([mapy_front, mapy_rear, mapy_left, mapy_right])

            if self.scene == 'avm':
                pass

        if self.project == 'mona':
            if self.scene == 'front_stereo':
                pass
            if self.scene == 'side':
                pass
            if self.scene == 'fisheye':
                pass
        
        print('cam: {}, mapx.shape: {}, mapy.shape: {}'.format(cam, mapx_out.shape, mapy_out.shape))
        self.scene2map[self.scene] = [mapx_out, mapy_out]

            
