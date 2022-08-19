import os
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1

import detect_image

ROOT = os.getcwd() # Đường dẫn của file


def data_loader(path_loader) -> dict:
	''' Tạo biến kiểu dữ liệu dict để chứa tên người, label và tên ảnh từng dữ liệu của người đó
	 Return data = {names_person : [label , [name_image1, name_image2, ...]]}
	'''
	path_data, label = {}, 0

	os.chdir(ROOT+ path_loader)
	names_person = os.listdir() 
	
	for name in names_person:
		os.chdir(ROOT+ path_loader + '/' + name)
		all_name_images = os.listdir() 
		path_data[name] = [label, all_name_images]
		label += 1
	os.chdir(ROOT)

	return path_data



def save_crop_face(path=ROOT+'/runs/crops') -> None:
	''' Xóa thư mục ảnh và crop từng khuôn mặt trên ảnh trong Person_data
		và lưu trữ tại runs/crops/
	'''
	path_data = data_loader('/Person_data') # Load thư mục 

	# Delete folders data trong runs/crops/
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



def search_center():
	''' Tìm tất cả các khoảng cách và vector trung bình của từng class
	'''
	path_data = data_loader('/runs/crops')
	average_knn = {}
	resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

	for name in path_data.keys():
		face_list = [] # Data đã được trích xuất đặc trưng
		for name_image in path_data[name][1]:
			img = cv2.imread(ROOT + '/runs/crops/' +name +'/' +name_image)
			img_tensor = change_type(img, PIL=True)
			out = resnet(img_tensor).detach()
			face_list.append(out)

		faces_tensor = torch.cat(face_list, dim=0) # Change to tensor
		average_knn[name] = torch.sum(faces_tensor, dim=0).div(len(path_data[name][1]))

	torch.save(average_knn, ROOT + '/Model_Parameter/KNN/Centers_KNN.pth')



def embedding_data():
	''' Save khuôn mặt crop đã trích đặc trưng 
	'''
	resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

	dataset = datasets.ImageFolder('runs/crops') 
	idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} 

	name_list = [] 
	embedding_list = [] 

	for img_crop, idx in dataset:
		face = change_type(img_crop)    
		emb = resnet(face).detach() 
		embedding_list.append(emb.detach()) 
		name_list.append(idx_to_class[idx])

	data = [embedding_list, name_list]
	torch.save(data, ROOT + '/Model_Parameter/Embedding/Embedding.pt') 



def change_type(face, PIL=False):
	''' Chuyển dạng dữ liệu đề trích đặc trưng bằng InceptionResnet
	'''
	if PIL:
		face_cv2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # Chuyển về ảnh RGB
		face_PIL = Image.fromarray(face_cv2) # Chuyển về định dang PIL
		resize_img = face_PIL.resize((240, 240))
	else:
		resize_img = face.resize((240, 240))
	face_tensor = torchvision.transforms.functional.pil_to_tensor(resize_img)
	processed_tensor = (face_tensor - 127.5) / 128.0
	face = processed_tensor.unsqueeze(0)
	return face



if __name__ == '__main__':
	save_crop_face()
	#embedding_data()
	#search_center()