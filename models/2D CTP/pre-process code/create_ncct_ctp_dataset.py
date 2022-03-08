
import joblib

import pyelastix
import os
from PIL import Image
# Get params and change a few values
params = pyelastix.get_default_params()
params.MaximumNumberOfIterations = 300
params.FinalGridSpacingInVoxels = 10

import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import sys
from skimage import transform, color

from tqdm import tqdm

from brainCT_utils import cv_registration, load_dicom_img, \
						load_dicom_img_three_window_levels, \
						load_dicom_ncct, \
						load_dicom_img_three_window_levels_1slice_3windows,\
						get_lesion_map_roi, get_ncct_roi

# import numpy as np
# from PIL import Image

def get_mask(mask_path):
	# input: mask_path
	# output: multi class mask
	arr_ms = get_raw_mask(mask_path)
	core = extract_core(arr_ms)
	pneu = extract_pneu(arr_ms, core)
	mask = create_multi_targets(core, pneu)
	return mask # float32

def get_raw_mask(mask_path):
	arr_ms = Image.open(mask_path)
	arr_ms = np.array(arr_ms)
	return arr_ms
	
def extract_core(arr_im):
	im_core = np.zeros_like(arr_im)
	im_core_mask = arr_im > 230
	im_core[im_core_mask] = arr_im[im_core_mask]
	im_core = np.uint8(im_core)
	return im_core

def extract_pneu(arr_im, im_core):
	im_pneu = np.maximum(0., arr_im - im_core)
	im_pneu = np.where(im_pneu > 50., 255., 0.)
	im_pneu = np.uint8(im_pneu)
	return im_pneu
	
def extract_background(arr_im):
#     im_arr = get_raw_mask(mask_path)
	im_bg = arr_im.copy()
	im_bg[im_bg > 10.] = 100.
	im_bg[im_bg < 10.] = 255.0
	im_bg[im_bg == 100.0] = 0.0
	im_bg = np.uint8(im_bg)
	return im_bg

def create_multi_targets(core, pneu):
	return np.asarray([core, pneu]).astype(np.float32)

def create_8bit_mask(bg, core, pneu):
	# bg: background
	mask = np.zeros_like(core)
	mask[bg == 255] = 0
	mask[core == 255] = 1
	mask[pneu == 255] = 2
	return np.uint8(mask)

def mask_extractor(lesion_map):
	infarction_core = lesion_map[:,:,0] - lesion_map[:,:,1]
	penumbra = lesion_map[:,:,1] - lesion_map[:,:,0]
	infarction_core[infarction_core<255] = 0
	penumbra[penumbra <255] = 0
	mask = penumbra//255* 127 + 255* (infarction_core//255)
	return mask.astype(np.uint8)

def penumbra_core_mask_extractor(lesion_map):
	mask = mask_extractor(lesion_map)
	mask[mask > 0] = 255 #mask.max() # combine penumbra + core = acute ischemic stroke lesion
	return mask

def make_dir(path):
	if not os.path.isdir(path):
		try:
			os.makedirs(path)
		except OSError:
			print ("Creation of the directory %s failed" % path)           

def register_ctp_ncct(im1, im2):
	'''
    inputs
		im1: NCCT moving
		im2: CTP fix
    return
        transformed NCCT
	'''
	assert (im1.shape==(512, 512))
	assert (im2.shape==(320, 320, 3))
#     im1 = im1[:, :, 1].astype(float) # brain channel
	im2 = im2[:, :, 2].astype(float)
	im1 = cv2.resize(im1, (im2.shape[:2]))

	im1_deformed, field = pyelastix.register(im1, im2, params, verbose=0) # im1/NCCT moving
	im2_d = np.array(im1_deformed)
	im2_d = np.clip(im2_d, 0, 255)
	return im2_d.astype(np.uint8) 

def ctp_register_to_ncct_3channels(im1ct, im2ctp):
	'''
	moving NCCT, fix CTP
		im1ct: NCCT 
		im2ctp: CTP 
	'''
	assert (im1ct.shape==(512, 512, 3))
	assert (im2ctp.shape==(320, 320, 3))
	res = []
	for i in range(3):
		im1 = im1ct[:, :, i].astype(float) # brain channel
		im2 = im2ctp[:, :, 2].astype(float)
		im1 = cv2.resize(im1, (im2.shape[:2]))

		im1_deformed, field = pyelastix.register(im1, im2, params, verbose=0) # moving im1/NCCT
	#     im1_deformed, field = pyelastix.register(im1, im2, params, verbose=0) # fix im2/CTP
		im2_d = np.array(im1_deformed)
		im2_d = np.clip(im2_d, 0, 255)
		res.append(im2_d.astype(np.uint8))
	res = np.array(res)
	res = res.transpose(1, 2, 0)
#     print('\nres: ', res.shape)
	return res


def save_img(fname, img):
	cv2.imwrite(fname, img)

def prepare_and_save(ncct_file, ctp_file, mask_path='', image_path=''):
	make_dir(mask_path)
	make_dir(image_path)

	img = load_dicom_img_three_window_levels_1slice_3windows(ncct_file) # img = 1 channel with 3 window levels
# 	img = load_dicom_img_three_window_levels(ncct_file) # img = 3 channel with 3 window levels
	roi_lesion, img_lesion = get_lesion_map_roi(ctp_file) 
	mask = penumbra_core_mask_extractor(roi_lesion)
	
# 	pid, accno  = ncct_file.split('/')[5], ncct_file.split('/')[6]
	mask_fname = ncct_file.split('/')[-1].replace('.dcm','_mask.png') #pid + '_' + accno + '_'+ 
	save_img(os.path.join(mask_path, mask_fname), mask)

	reg = register_ctp_ncct(img, img_lesion)
# 	reg = ctp_register_to_ncct_3channels(img, img_lesion)
	fname =  ncct_file.split('/')[-1].replace('dcm','png') # pid + '_' + accno + '_'+
	save_img(os.path.join(image_path, fname), reg)

def prepare_images_njobs(df, n_jobs=-1):
	joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(prepare_and_save)(r.NCCT, r.Lesion_Map, mask_path, image_path) for ix, r in tqdm(df.iterrows())) 

if __name__ == '__main__':
    # setting paths to save images and masks
    mask_path = 'mask'
    image_path = 'image'
    df_all = pd.read_csv('ncct_lesion_map.csv') # NCCT and CTP pairs
    prepare_images_njobs(df_all)