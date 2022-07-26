import os
from PIL import Image
import cv2
import numpy as np
import time

import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

import images, videos
from general import load_model, average_KNN, all_KNN

ROOT = os.getcwd()



class YOLOV5():
	''' YOLOV5 load file detect_image từ file YoloV5 để tách và phân loại đối tượng
		Ngoài ra crop từng object face trong ảnh để qua bước nhận diện 
	'''
	def __init__(self, source, KNN='all'):
		self.source = source # Đường dẫn của hình ảnh dùng để test
		self.KNN = KNN

	def take_object(self):
		# Lấy các thông tin của từng object trong ảnh là center x,y,x,y và id class từ model YOLOV5
		if self.source != 0 and self.source != 1:
			output = images.run(
				weights=ROOT + '/models/parameter/YOLOv5/weight_train.pt',
				source= ROOT + self.source, # file/dir/URL/glob, 0 for webcam
				data=ROOT + '/data/facemask.yaml',
				imgsz=(640, 640),  # inference size (height, width)
				conf_thres=0.25,  # confidence threshold
				iou_thres=0.45,  # NMS IOU threshold
				max_det=1000,
				save_crop=False,  # save cropped prediction boxes
				view_img=False,  # maximum detections per image
				project=ROOT + '/detect',  # save results to project/name
				name='Face_Mask',  # save results to project/name
				)

			return output

		else:
			videos.run(
				weights=ROOT + '/models/parameter/YOLOv5/weight_train.pt',
				source= 0, # file/dir/URL/glob, 0 for webcam
				data=ROOT + '/data/facemask.yaml',
				imgsz=(640, 640),  # inference size (height, width)
				conf_thres=0.25,  # confidence threshold
				iou_thres=0.45,  # NMS IOU threshold
				max_det=1000,
				save_crop=False,  # save cropped prediction boxes
				view_img=False,  # maximum detections per image
				project=ROOT + '/detect',  # save results to project/name
				name='Face_Mask',  # save results to project/name
				KNN=self.KNN,
				)
		

	def detect_box(self, xyxy, im: np.array, gain=1.02, pad=10, BGR=True):
		# Crop object từ ảnh cần nhận dạng
		xyxy = torch.tensor(xyxy).view(-1, 4)
		b = self.xyxy2xywh(xyxy)  # boxes
		b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
		xyxy = self.xywh2xyxy(b).long()
		self.clip_coords(xyxy, im.shape)
		crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
		return crop


	def xyxy2xywh(self, x):
		# Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
		y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
		y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
		y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
		y[:, 2] = x[:, 2] - x[:, 0]  # width
		y[:, 3] = x[:, 3] - x[:, 1]  # height
		return y


	def xywh2xyxy(self, x):
		# Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
		y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
		y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
		y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
		y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
		y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
		return y


	def clip_coords(self, boxes, shape):
		# Clip bounding xyxy bounding boxes to image shape (height, width)
		if isinstance(boxes, torch.Tensor):  # faster individually
			boxes[:, 0].clamp_(0, shape[1])  # x1
			boxes[:, 1].clamp_(0, shape[0])  # y1
			boxes[:, 2].clamp_(0, shape[1])  # x2
			boxes[:, 3].clamp_(0, shape[0])  # y2
		else:  # np.array (faster grouped)
			boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
			boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


yolo = YOLOV5(source='/test/')  # Đường dẫn của hình ảnh dùng để test
resnet = InceptionResnetV1(pretrained='vggface2').eval() 
#vgg_transfer = torch.load(ROOT + '/models/parameter/VGG/model_transfer.pt', map_location='cpu')


def identify_images(KNN, view_result, save_image):
	''' Liên kết các model lại với nhau để ra kết quả sau cùng
		Var -view_result: Demo kết quả
			-save_image: Lưu lại kết quả
	'''
	resnet, emb_face = load_model(KNN=KNN)
	data = yolo.take_object() # Detect Object, type return = dict
	paths = data.keys() # Load đường dẫn của tất cả hình ảnh test
	colors = {
		'Not Mask': (0, 0, 255),
		'Correct': (0, 153, 0),
		'Incorrect': (0, 255, 255),
		} # Màu sắc của class

	# Xóa data trong file nơi lưu hình ảnh đã nhận diện
	if save_image:
		name = 0
		os.chdir(ROOT + '/detect')
		arr = os.listdir()
		for char in arr:
			if char[-4:] == '.jpg':
				os.remove(char)
		os.chdir(ROOT)

	# Nhận diện với từng bức ảnh
	for path in paths:
		start_time = time.time()	    
		im = cv2.imread(path)
		im0 = im.copy()

		n = len(data[path]) # Đếm số đối tượng trên mỗi bức ảnh

		# Nhận diện từng đối tượng trên bức ảnh
		print('Path: ', path)
		for i in range(n):
			img_crop = yolo.detect_box(data[path][i][1], im) # Ảnh khuôn mặt đã được cắt
			name_class = data[path][i][0] # 3 class phân loại người đeo khẩu trang

			if name_class == 'Not Mask':
				if KNN == 'all':
					result, distance = all_KNN(img_crop, model=resnet, emb_face=emb_face, threshold=0.6)
				else:
					result, distance = average_KNN(img_crop, model=resnet, emb_face=emb_face, threshold=0.6) # Nhận kết quả nhận diện khuôn mặt
				end_time = time.time()		
				print('Object is {}, distance is {}'.format(result, distance.detach().numpy()))
				print(end_time - start_time)
			elif name_class == 'Correct':
				result, distance = 'Vui long bo khau trang xuong', ''
				print('Object {} deo khau trang'.format(i+1))
			else:
				result, distance = 'Ban deo khau trang sai cach', ''
				print('Object {} deo khau trang sai canh'.format(i+1))

			# Open CV để hiển thị kết quả trên ảnh
			if view_result or save_image:
				# chuyển tọa độ từ xyxy về xywh
				x1, y1, x2, y2 = data[path][i][1]
				w, h = int(x2-x1), int(y2-y1)
				x, y = int(x1), int(y1)

				if im.shape[0]>im.shape[1]: size=im.shape[1]/800
				else: size=im.shape[0]/800

				if result == 'No name' or name_class != 'Not Mask':
					text = result
				else: 
					text = result+', '+str(distance.detach().numpy())
				cv2.putText(im0, text, (x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=size, color=colors[name_class])
				cv2.putText(im0, 'Object {}: '.format(i+1) + name_class, (x,y+h+int(h*0.1)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=size, color=colors[name_class])		
				cv2.rectangle(im0, (x,y), (x+w, y+h), colors[name_class], 5)

		# Demo kết quả
		if view_result:
			cv2.imshow('image', im0)
			cv2.waitKey(0)

		# Lưu lại kết quả
		if save_image:
			cv2.imwrite(ROOT+'/detect/'+str(name)+'.jpg', im0)
			name += 1


def identify_real_time(KNN):
	yolo = YOLOV5(source=0,
		KNN=KNN)
	yolo.take_object()


if __name__ == '__main__':
	# identify_images(
	# 	KNN='center', # Chọn phương pháp phân loại
	# 	view_result=False, # Demo kết quả
	# 	save_image=True # Lưu lại kết quả
	# 	)

	identify_real_time(
		KNN='center', # Chọn phương pháp phân loại
		)