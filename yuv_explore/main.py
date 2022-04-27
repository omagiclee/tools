import sys

import cv2
import numpy as np


class YuvExp:
    def __init__(self, yuv_size):
        self.imw = 1920
        self.imh = 1080
        self.yuv_size = (int(self.imh*1.5), self.imw)

    def resize(self, yuv_fpath, scale):
        ret, yuv = self.read_yuv(yuv_fpath)
        if not ret:
            return None
        
        Y = yuv[:self.imh, :]
        UV = yuv[self.imh:, :]
        # U = UV[::1, ::2]
        # V = UV[::1, 1::2]
        UV_new = UV.reshape(UV.shape[1], UV.shape[0] // 2, 2)
        UV_new = UV_new.transpose()
        UV_new = cv2.resize(UV_new, (UV.shape[1] * scale, UV.shape[0] * scale))
        UV_new = cv2.

        Y_new = cv2.resize(Y, (Y.shape[1] * scale, Y.shape[0] * scale))
        # U_new = cv2.resize(U, (U.shape[1] * scale, U.shape[0] * scale))
        # V_new = cv2.resize(V, (V.shape[1] * scale, V.shape[0] * scale))

        YUV_new = np.vstack((Y_new))
        
        RGB = cv2.cvtColor(YUV_new, cv2.COLOR_YUV2RGB_NV21)
        return RGB

    def roi(self, yuv_fpath):
        ret, yuv = self.read_yuv(yuv_fpath)
        if not ret:
            return None

        Y = yuv[:self.imh, :]
        UV = yuv[self.imh:, :]
        U = UV[::1, ::2]
        V = UV[::1, 1::2]

        Y_new = cv2.resize(Y, (self.imw // 2, self.imh // 2))
        YUV_middle = np.array((Y_new, U, V)).transpose(1, 2, 0) # (540, 960, 3)     
        
        # ROI
        ROI = YUV_middle[200:500, 500:900, :]
        Y_ROI = ROI[:, :, 0]
        UV_ROI = ROI[:, :, 1:]
        UV_ROI = cv2.resize(UV_ROI, (UV_ROI.shape[1] // 2, UV_ROI.shape[0] // 2))
        UV_ROI = UV_ROI.reshape(UV_ROI.shape[0], -1)
        YUV_ROI = np.vstack((Y_ROI, UV_ROI))

        RGB = cv2.cvtColor(YUV_ROI, cv2.COLOR_YUV2RGB_NV21)
        return RGB

    def crop(self):
        pass

    def read_yuv(self, yuv_fpath):
        try:
            with open(yuv_fpath, 'rb') as f:
                raw = f.read(self.yuv_size[0] * self.yuv_size[1])
                yuv = np.frombuffer(raw, dtype=np.uint8)
                yuv = yuv.reshape(self.yuv_size)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv


if __name__ == '__main__':
    # im = cv2.imread('frame_vc6_298.yuv')
    # print(im.shape)
    imfpath = 'frame_vc6_298.yuv'
    yuv_exp = YuvExp((int(1080*1.5), 1920))
    yuv, bgr = yuv_exp.resize(imfpath)

    cv2.imshow('view', bgr)
    key = cv2.waitKey(0)
    if key == ord('q'):
        sys.exit()
