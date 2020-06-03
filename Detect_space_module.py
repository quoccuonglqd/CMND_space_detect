import cv2
import numpy as np 
from os import listdir
import os.path as osp
import argparse
import time
from Config_class import Config

THRESHOLD_VALUES = ['adaptive_mean','adaptive_gaussian','normal']
CONTOUR_MODE_VALUES = {'external':cv2.RETR_EXTERNAL,
					   'list':cv2.RETR_LIST,
					   'ccomp':cv2.RETR_CCOMP,
					   'tree':cv2.RETR_TREE}
CONTOUR_METHOD_VALUES = {'none':cv2.CHAIN_APPROX_NONE,
					   'simple':cv2.CHAIN_APPROX_SIMPLE,
					   'tc89_l1':cv2.CHAIN_APPROX_TC89_L1,
					   'tc89_kcos':cv2.CHAIN_APPROX_TC89_KCOS}


def Convert_to_binary(image,blur_kernel_size,threshold_type,threshold_argument):
	r'''Convert a color image to binary
	    Args:
	        image: numpy array represent color image
	        blur_kernel_size:size of kernel used to blur the image
	        threshold_type: got value in ('adaptive_mean','adaptive_gaussian','normal'), the method used to get the binary image
	        threshold_argument: argument for threshold method
	                            int if threshold_type == 'normal'
	                            tuple2 in other cases
	    '''

	blur = cv2.GaussianBlur(image, (blur_kernel_size,blur_kernel_size),0)
	gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
	
	# gray = cv2.equalizeHist(gray)
	
	if threshold_type == 'adaptive_mean':
		return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_argument[0], threshold_argument[1])
	elif threshold_type == 'adaptive_gaussian':
		return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_argument[0], threshold_argument[1])
	elif threshold_type == 'normal':
		_,thresh = cv2.threshold(gray,np.amin(gray) + threshold_argument, 255, cv2.THRESH_BINARY)
		return thresh 

def Check_contour(contour, image_height, image_width ,y_ratio = 0.6, x_ratio = 0.2):
	r'''Check if a contour is valid, not too small or too big
	    Args:
	        contour: ndarray of shape (n,1,2)
	        image_height: int
	        image_width: int
	        y_ratio: float
	        x_ratio: float
	'''

	[x,y,w,h] = cv2.boundingRect(contour)
	if h>300 and w>300:
		return False
	# discard areas that are too large
	if h < y_ratio*image_height or w < x_ratio*image_width:
		return False
	# discard areas that are too small
	return True 

def Non_eroded_process(image,mode,method,x_ratio,y_ratio,contour_limit):
	r'''If the image before being eroded has exactly 2 contours, return it
	    Otherwise, remove noise in the image
	    Args:
	        image: numpy array represent binary image
	        mode: str, mode of findcontour function
	        method: str, method of findcontour function
	        x_ratio: float
	        y_ratio: float
	        contour_limit: int, the limit of contour, if its size is smaller, it will be removed
	    Return: tuple2 of processed image and (None if has not answer for now or result is yes)
	'''
	  
	image_ = image.copy()
	if mode not in CONTOUR_MODE_VALUES or method not in CONTOUR_METHOD_VALUES:
		return (image_,None)
	contours, hierarchy = cv2.findContours(image,CONTOUR_MODE_VALUES[mode],CONTOUR_METHOD_VALUES[method])
	if (len(contours)) == 2 and Check_contour(contours[0],image.shape[0],image.shape[1],x_ratio,y_ratio) and Check_contour(contours[1],image.shape[0],image.shape[1],x_ratio,y_ratio) and min(np.max(contours[0][:,0,0]),np.max(contours[1][:,0,0])) < 0.75*image.shape[1]: 
		return (image_,min(np.max(contours[0][:,0,0]),np.max(contours[1][:,0,0])))
	else:
		for i in range(len(contours)):
			if len(contours[i])<contour_limit:
				cv2.drawContours(image_,contours,i,color=255,thickness=cv2.FILLED)
		return (image_,None)

def Eroded_detect(image,mode,method,size,number_iterations,x_ratio,y_ratio):
	r'''Erode image and find result
	    Args:
	        image: numpy array represent binary image
	        mode: str, mode of findcontour function
	        method: str, method of findcontour function
	        size: int,erode kernel size
	        number_iterations:int, number of iterations to make erosion
	        x_ratio: float
	        y_ratio: float
	'''

	kernel = np.ones((size,size),np.uint8)
	eroded = cv2.erode(image,kernel,iterations = number_iterations)
	eroded = 255 - eroded	
	contours, hierarchy = cv2.findContours(eroded,CONTOUR_MODE_VALUES[mode],CONTOUR_METHOD_VALUES[method])

	ls = []
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		if Check_contour(contour,image.shape[0],image.shape[1],x_ratio,y_ratio)==False:
			continue
		ls.append(x+w)
		'''cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
		roi = image[y:y + h, x:x + w]
		cv2.imshow('img',roi)
		cv2.waitKey(0)'''
	if len(ls) >= 3 or len(ls)==0 or (len(ls)==1 and ls[0]>=0.75*image.shape[1])  or min(ls)>=0.75*image.shape[1]:
		return None
	else:
	    return min(ls)

def Process_in_single_cfg(path,cfg,is_visualize = False):
	r'''Full process in single config
	    Args:
	        path:str, path to a single image
	        cfg: class Config 
	        is_visualize:Boolen,whether or not visualization
	'''

	image = cv2.imread(path)
	thresh = Convert_to_binary(image,cfg.blur_kernel_size,cfg.threshold_type,cfg.threshold_argument)
	image_,res = Non_eroded_process(thresh,cfg.mode[0],cfg.method[0],cfg.x_ratio,cfg.y_ratio,cfg.contour_limit)
	if (res!=None):
		img = image
		cv2.line(img, (res,0), (res,img.shape[0]), (255,255,0), thickness=4, lineType=8, shift=0)
		if is_visualize == True:
			cv2.imshow('result',img)
			cv2.waitKey(0)
		return res
	res = Eroded_detect(image_,cfg.mode[1],cfg.method[1],cfg.erode_kernel_size,cfg.number_iterations,cfg.x_ratio,cfg.y_ratio)
	if (res!=None):
		img = image
		cv2.line(img, (res,0), (res,img.shape[0]), (255,255,0), thickness=4, lineType=8, shift=0)
		if is_visualize == True:
			cv2.imshow('result',img)
			cv2.waitKey(0)
		return res

def Process_in_multiple_cfg(path,cfg_ls=['Configs/config1.py','Configs/config2.py','Configs/config3.py'],is_visualize=False):
	r'''Full process in multiple config
	    Args:
	        path:str, path to a single image
	        cfg_ls: list of class Config 
	        is_visualize:Boolen,whether or not visualization
	'''

	ls = []
	for cfg in cfg_ls:
		cfg = Config(cfg)
		res = Process_in_single_cfg(path,cfg)
		if res != None:
			ls.append(res)
	if len(ls)>0:
		if is_visualize == True:
			img = cv2.imread(path)
			cv2.line(img, (int(sum(ls)/len(ls)),0), (int(sum(ls)/len(ls)),img.shape[0]), (255,255,0), thickness=4, lineType=8, shift=0)
			cv2.imshow('result',img)
			cv2.waitKey(0)
		return int(sum(ls)/len(ls))
	else:
		return None