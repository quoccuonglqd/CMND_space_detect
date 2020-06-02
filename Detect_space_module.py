import cv2
import numpy as np 
from os import listdir
import os.path as osp
import argparse
import time

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
	blur = cv2.GaussianBlur(image, (11,11),0)
	gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
	#gray = cv2.equalizeHist(gray)
	if threshold_type == 'adaptive_mean':
		return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_argument[0], threshold_argument[1])
	elif threshold_type == 'adaptive_gaussian':
		return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_argument[0], threshold_argument[1])
	elif threshold_type == 'normal':
		_,thresh = cv2.threshold(gray,np.amin(gray) + threshold_argument, 255, cv2.THRESH_BINARY)
		return thresh 

def Check_contour(contour, image_height, image_width ,y_ratio = 0.6, x_ratio = 0.2):
	[x,y,w,h] = cv2.boundingRect(contour)
	if h>300 and w>300:
		return False
	# discard areas that are too large
	if h < y_ratio*image_height or w < x_ratio*image_width:
		return False
	# discard areas that are too small
	return True 

def Non_eroded_process(image,mode,method,x_ratio,y_ratio):
	image_ = image.copy()
	if mode not in CONTOUR_MODE_VALUES or method not in CONTOUR_METHOD_VALUES:
		return (image_,None)
	contours, hierarchy = cv2.findContours(image,CONTOUR_MODE_VALUES[mode],CONTOUR_METHOD_VALUES[method])
	if (len(contours)) == 2 and Check_contour(contours[0],image.shape[0],image.shape[1],x_ratio,y_ratio) and Check_contour(contours[1],image.shape[0],image.shape[1],x_ratio,y_ratio):
		return (image_,min(np.max(contours[0][:,0,1]),np.max(contours[1][:,0,1])))
	else:
		for i in range(len(contours)):
			if len(contours[i])<4:
				cv2.drawContours(image_,contours,i,color=255,thickness=cv2.FILLED)
		return (image_,None)

def Eroded_detect(image,mode,method,size,number_iterations,x_ratio,y_ratio):
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
		#cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
		#roi = image[y:y + h, x:x + w]
		#cv2.imshow('img',roi)
		#cv2.waitKey(0)
	if len(ls) >= 3 or len(ls)==0 or (len(ls)==1 and ls[0]>=0.9*image.shape[1])  or min(ls)>=0.75*image.shape[1]:
		return None
	else:
	    '''img = image
	    cv2.line(img, (min(ls),0), (min(ls),img.shape[0]), (255,255,0), thickness=4, lineType=8, shift=0)
	    cv2.imshow('result1',img)'''
	    return min(ls)

def Process(path,cfg):
	start_time = time.time()
	image = cv2.imread(path)
	thresh = Convert_to_binary(image,cfg.blur_kernel_size,cfg.threshold_type,cfg.threshold_argument)
	image_,res = Non_eroded_process(thresh,cfg.mode[0],cfg.method[0],cfg.x_ratio,cfg.y_ratio)
	if (res!=None):
		img = image
		cv2.line(img, (res,0), (res,img.shape[0]), (255,255,0), thickness=4, lineType=8, shift=0)
		cv2.imshow('result1',img)
		cv2.waitKey(0)
		return
	res = Eroded_detect(image_,cfg.mode[1],cfg.method[1],cfg.erode_kernel_size,cfg.number_iterations,cfg.x_ratio,cfg.y_ratio)
	print(time.time()-start_time)
	if (res!=None):
		img = image
		cv2.line(img, (res,0), (res,img.shape[0]), (255,255,0), thickness=4, lineType=8, shift=0)
		cv2.imshow('result1',img)
		cv2.waitKey(0)
		return
