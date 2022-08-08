import os
from PIL import Image

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

import detect_image

ROOT = os.getcwd() # Đường dẫn của file



def data_loader() -> dict: 
	# Return data = {names_person : [label , [name_image1, name_image2, ...]]}
	path_data, label = {}, 0

	os.chdir(ROOT+ '/Person_data')
	names_person = os.listdir() 
	
	for name in names_person:
		os.chdir(ROOT+ '/Person_data/'+ name)
		all_name_images = os.listdir() 
		path_data[name] = [label, all_name_images]
		label += 1
	os.chdir(ROOT)

	return path_data
path_data = data_loader()



def train_feature(features=8192): # Feature: Số đặc trưng đầu ra sau khi qua mô hình Resnet
	''' Hàm áp dụng mô hình có sẵn của facenet-pytorch hỗ trợ và model yolo
		Detect Object yolov5
		Feature Extraction bằng pre-train với bộ weight của vggface2
		Số chiều feature được trích ra là 512
	'''
	resnet = InceptionResnetV1(pretrained='vggface2').eval() # Chọn mô hình pre-train

	save_crop_face() # Crop and save with yolo

	feature_list = [] # Chứa các đặc trưng đầu ra
	name_list = [] # Chứa label và name của đối tượng

	# Trích feature cho mỗi ảnh được crop 
	for names_person in path_data.keys():
		for name_image in path_data[names_person][1]:
			img_crop = cv2.imread(ROOT + '/runs/crops/' + names_person +'/'+ name_image)
			img_resize = cv2.resize(img_crop, (240, 240))
			data_tensor = torch.from_numpy(img_resize) #Change to tensor
			data_tensor = data_tensor.unsqueeze(0) #Flatten
			face = data_tensor.permute(0, 3, 1, 2).type(torch.float32) #Custom tensor để đầu vào phù hợp với resnet
			feature = resnet(face) 

			feature_list.append(feature[0].detach().numpy()) 
			name_list.append([path_data[names_person][0], names_person]) 

	data = [feature_list, name_list]
	torch.save(data, 'Model_Parameter/FaceNet/Face_data02.pt') # saving Face_data.pt file



def save_crop_face(path=ROOT+'/runs/crops'):
	# Delete folders data in path
	os.chdir(path)
	folders = os.listdir()
	for folder in folders:
		os.chdir(path + '/' + folder)
		files = os.listdir()

		for file in files:
		    os.remove(file)
		os.chdir(path)
		os.rmdir(folder)
	os.chdir(ROOT)

	# Crop face bằng yolo trong file detect_image
	for name in path_data.keys():
		detect_image.run(
			weights=ROOT + '/Model_Parameter/YOLOv5/weight_train.pt',
			source=ROOT + '/Person_data/' + name +'/', # file/dir/URL/glob, 0 for webcam
			data=ROOT + '/data/facemask.yaml',
			imgsz=(640, 640),  # inference size (height, width)
			conf_thres=0.25,  # confidence threshold
			iou_thres=0.45,  # NMS IOU threshold
			max_det=1000,
			save_crop=True,  # save cropped prediction boxes
			view_img=False,  # maximum detections per image
			project=ROOT + '/runs/crops',  # save results to project/name
			name=name, # Name of persons
			)



def train_classify(path= 'Model_Parameter/FaceNet/Face_data02.pt', C=[0.001, 0.01, 0.1, 1, 10, 100]):
	''' Sử dụng Soft SVM để huấn luyện và phân loại
	'''
	saved_data = torch.load(path) # Load data đã được trích feature 
	X_train = np.array(saved_data[0]) # Convert to numpy

	y_train = [] # Khởi tạo list Label để train
	for i in range(len(X_train)):
		y_train.append(int(saved_data[1][i][0])) # Chuyển label là tên person về số
	y_train = np.array(y_train) # Convert to numpy

	for c in C:
		model = SVC(kernel='linear', C=c, class_weight='balanced')
		clf = OneVsRestClassifier(model) 
		clf.fit(X_train, y_train)


		result = clf.predict(X_train)
		count = 0 # Biến đếm số lượng dự đoán để tính accuracy
		n = len(y_train)
		for i in range(n):
			if result[i] == y_train[i]:
				count += 1

		print("Accuracy of train data: ",(count/n)*100,"%")

		# for name in arr:
		# 	img = cv2.imread(ROOT + '/Images_test/' + name)
		# 	face, prob = mtcnn(img, return_prob=True)
		# 	if face is not None and (prob > 0.90): # if face detected and porbability > 90%
		# 		emb = resnet(face.unsqueeze(0))
		# 	result = clf.predict(emb[0].detach().numpy())
		# 	for name in idx_to_class.keys():
		# 		if idx_to_class[name] == result[0]:
		# 			print(name)

if __name__ == '__main__':
	train_feature()
