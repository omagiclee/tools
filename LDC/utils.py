from pathlib import Path
import json

import numpy as np
import xml.etree.ElementTree as ET
import yaml


def load_intrinsic_paras(para_p, mode):
    if not Path(para_p).is_file():
        assert(False)
    
    cam_intrinsics_map = {}
    if mode == 'saic':
        tree = ET.parse(para_p)
        root = tree.getroot()
        spd = root.find('spd')
        cam_intrinsics_map['left_front'] = get_K_D(spd, 'nearside_mirror', mode)
        cam_intrinsics_map['left_rear'] = get_K_D(spd, 'nearside_wing', mode)
        cam_intrinsics_map['right_front'] = get_K_D(spd, 'offside_mirror', mode)
        cam_intrinsics_map['right_rear'] = get_K_D(spd, 'offside_wing', mode)
    elif mode == 'chery':
        tree = ET.parse(para_p)
        root = tree.getroot()
        paras_fov120 = root.find('camera_param').find('front_near')  
        paras_fov30 = root.find('camera_param').find('front_middle') 
        cam_intrinsics_map['fov120'] = get_K_D(paras_fov120, 'front_near', mode)
        cam_intrinsics_map['fov30'] = get_K_D(paras_fov30, 'front_middle', mode)
    elif para_p.endswith('yaml'):  # yaml
        with open(para_p, 'r') as f:
            intrinsics = yaml.load(f, Loader=yaml.Loader)
            cam_intrinsics_map['left_front'] = get_K_D(intrinsics, 'left_front', mode='dict')
            cam_intrinsics_map['left_rear'] = get_K_D(intrinsics, 'left_rear', mode='dict')
            cam_intrinsics_map['right_front'] = get_K_D(intrinsics, 'right_front', mode='dict')
            cam_intrinsics_map['right_rear'] = get_K_D(intrinsics, 'right_rear', mode='dict')
    else:
        assert(False)

    return cam_intrinsics_map

def get_K_D(paras, cam, mode):
    if mode == 'saic':
        fx = eval(paras.find('{}_fx'.format(cam)).text)
        fy = eval(paras.find('{}_fy'.format(cam)).text)
        cx = eval(paras.find('{}_cx'.format(cam)).text)
        cy = eval(paras.find('{}_cy'.format(cam)).text)
        k1 = eval(paras.find('{}_k1'.format(cam)).text)
        k2 = eval(paras.find('{}_k2'.format(cam)).text)
        k3 = eval(paras.find('{}_k3'.format(cam)).text)
        k4 = eval(paras.find('{}_k4'.format(cam)).text)
        k5 = eval(paras.find('{}_k5'.format(cam)).text)
        k6 = eval(paras.find('{}_k6'.format(cam)).text)
        p1 = eval(paras.find('{}_p1'.format(cam)).text)
        p2 = eval(paras.find('{}_p2'.format(cam)).text)
    elif mode == 'chery':
        fx = eval(paras.find('fx'.format(cam)).text)
        fy = eval(paras.find('fy'.format(cam)).text)
        cx = eval(paras.find('cx'.format(cam)).text)
        cy = eval(paras.find('cy'.format(cam)).text)
        k1 = eval(paras.find('k1'.format(cam)).text)
        k2 = eval(paras.find('k2'.format(cam)).text)
        k3 = eval(paras.find('k3'.format(cam)).text)
        k4 = eval(paras.find('k4'.format(cam)).text)
        k5 = eval(paras.find('k5'.format(cam)).text)
        k6 = eval(paras.find('k6'.format(cam)).text)
        p1 = eval(paras.find('p1'.format(cam)).text)
        p2 = eval(paras.find('p2'.format(cam)).text)
    elif mode == 'dict':
        fx = paras[cam]['fx']
        fy = paras[cam]['fy']
        cx = paras[cam]['cx']
        cy = paras[cam]['cy']
        k1 = paras[cam]['k1']
        k2 = paras[cam]['k2']
        k3 = paras[cam]['k3']
        k4 = paras[cam]['k4']
        k5 = paras[cam]['k5']
        k6 = paras[cam]['k6']
        p1 = paras[cam]['p1']
        p2 = paras[cam]['p2']
    else:
        assert(False)

    mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
    return mtx, dist

def load_extrinsic_paras(extrinsic_fpath):
    if not Path(extrinsic_fpath).is_file():
        assert(False)

    extrinsics = {}
    if extrinsic_fpath.endswith('.xml'):  # xml
        tree = ET.parse(extrinsic_fpath)
        root = tree.getroot()
        spd = root.find('spd')   
        extrinsics['left_front'] = get_R_T(spd, 'nearside_mirror', mode='xml')
        extrinsics['left_rear'] = get_R_T(spd, 'nearside_wing', mode='xml')
        extrinsics['right_front'] = get_R_T(spd, 'offside_mirror', mode='xml')
        extrinsics['right_rear'] = get_R_T(spd, 'offside_wing', mode='xml')
    else:
        assert(False)

def get_R_T(self, extrinsics, cam_alias, mode):
    R = []
    if mode == 'xml':
        for i in range(0, 9):
            R.append(eval(extrinsics.find('{}_rotate_matrix_element_{}'.format(cam_alias, i)).text))

    R = np.array(R).reshape(3, 3)
    T = np.array([0, 0, 0])
    return R, T

def load_yaml(para_p):
    paras = {}
    with open(para_p, 'r') as f:
        paras = yaml.load(f, Loader=yaml.Loader)
    
    return paras

def load_json(para_p):
    paras = json.load(open(para_p, 'r'))
    
    return paras

def save_ldc_map(cam_undistort_map, savep):
    def FillMiddle(ad_offset, offset, height, width):
        ad_offset[int(height / 2), :int(width / 2)] = (offset[int(height / 2), :int(width / 2)] + \
                                                    offset[int(height / 2) - 1, :int(width / 2)]) / 2.0
        ad_offset[int(height / 2), int(width / 2) + 1:] = (offset[int(height / 2), int(width / 2):] + \
                                                        offset[int(height / 2) - 1, int(width / 2):]) / 2.0
        ad_offset[:int(height / 2), int(width / 2)] = (offset[:int(height / 2), int(width / 2)] + \
                                                    offset[:int(height / 2), int(width / 2) - 1]) / 2.0
        ad_offset[int(height / 2) + 1:, int(width / 2)] = (offset[int(height / 2):, int(width / 2)] + \
                                                        offset[int(height / 2):, int(width / 2) - 1]) / 2.0
        ad_offset[int(height / 2), int(width / 2)] = (ad_offset[int(height / 2) - 1, int(width / 2)] + \
                                                    ad_offset[int(height / 2) + 1, int(width / 2)] + \
                                                    ad_offset[int(height / 2), int(width / 2) - 1] + \
                                                    ad_offset[int(height / 2), int(width / 2) + 1]) / 4.0
    
    savep = savep / 'maps'
    if not Path(savep).exists():
        Path(savep).mkdir(parents=True, exist_ok=True)

    m = 0
    m_step = 2 ** m
    for cam in cam_undistort_map:
        map1, map2 = cam_undistort_map[cam]

        map_savefp = '{}/remap.txt'.format(savep)

        with open(map_savefp, 'w+') as fp:
            offset_x = np.zeros(map1.shape)
            offset_y = np.zeros(map2.shape)
            height, width = map1.shape
            for i in range(height):
                for j in range(width):
                    offset_x[i, j] = (map1[i, j] - j) * 8
                    offset_y[i, j] = (map2[i, j] - i) * 8

            ad_offset_x = np.zeros((map1.shape[0] + 1, map1.shape[1] + 1))
            ad_offset_y = np.zeros((map2.shape[0] + 1, map2.shape[1] + 1))
            # fill adjust offset
            # 1st Quadrant
            ad_offset_x[:int(height / 2), int(width / 2) + 1:] = offset_x[:int(height / 2), int(width / 2):]
            ad_offset_y[:int(height / 2), int(width / 2) + 1:] = offset_y[:int(height / 2), int(width / 2):]

            # 2nd Quadrant
            ad_offset_x[:int(height / 2), :int(width / 2)] = offset_x[:int(height / 2), :int(width / 2)]
            ad_offset_y[:int(height / 2), :int(width / 2)] = offset_y[:int(height / 2), :int(width / 2)]

            # 3rd Quadrant
            ad_offset_x[int(height / 2) + 1:, :int(width / 2)] = offset_x[int(height / 2):, :int(width / 2)]
            ad_offset_y[int(height / 2) + 1:, :int(width / 2)] = offset_y[int(height / 2):, :int(width / 2)]

            # 4th Quadrant
            ad_offset_x[int(height / 2) + 1:, int(width / 2) + 1:] = offset_x[int(height / 2):, int(width / 2):]
            ad_offset_y[int(height / 2) + 1:, int(width / 2) + 1:] = offset_y[int(height / 2):, int(width / 2):]

            FillMiddle(ad_offset_x, offset_x, height, width)
            FillMiddle(ad_offset_y, offset_y, height, width)

            for i in range(0, height + 1, m_step):
                for j in range(0, width + 1, m_step):
                    fp.write(str(int(round(ad_offset_x[i, j]))) + ' ' + str(int(round(ad_offset_y[i, j]))) + '\n')
