import os
import PIL
from PIL import Image
import cv2
import numpy as np 
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

ROOT = os.getcwd()


def change_type(face, PIL=False):
	''' Chuyển dạng dữ liệu về dạng tensor
	'''
	if PIL:
		resize_img = face.resize((224, 224))
	else:
		face_cv2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # Chuyển về ảnh RGB
		face_PIL = Image.fromarray(face_cv2) # Chuyển về định dang PIL
		resize_img = face_PIL.resize((224, 224))
	face_tensor = torchvision.transforms.functional.pil_to_tensor(resize_img)
	processed_tensor = (face_tensor - 127.5) / 128.0
	face = processed_tensor.unsqueeze(0)
	return face


def load_model(KNN='all'):
	resnet = InceptionResnetV1(pretrained='vggface2').eval()
	if KNN == 'all':
		data = torch.load(ROOT + '/models/parameter/embedding/resnet_512.pt')
		return resnet, data
	elif KNN == 'center':
		data = torch.load(ROOT + '/models/parameter/KNN/resnet_512.pth')
		return resnet, data



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


def all_KNN(img_crop, model, emb_face, threshold=0.5):
	''' Nhận ảnh khuôn mặt đã được cắt và trả về nhận diện đối tượng và khoảng cách 
		từ ảnh tới các điểm dữ liệu trung bình
		Var -img_crop: ảnh khuôn mặt đã được cắt type = np.array
			-threshold: đặt ngưỡng nếu khoảng cách quá xa thì là class ko xác định
	'''
	embeddings_list = [] # Chứa tất cả khoảng cách của face tới với toàn bộ dữ liệu
	names_list = [] # Chứa tên tất cả các đối tượng
	face = change_type(img_crop)

	output = model(face)

	for index in range(len(emb_face[0])):
		euclidean_distance = F.pairwise_distance(output, emb_face[0][index])
		embeddings_list.append(euclidean_distance)
		names_list.append(emb_face[1][index])

	min_list = min(embeddings_list)
	if min_list > threshold:
		return 'No_name', min_list.double()
	else:
		return names_list[embeddings_list.index(min_list)], min_list.double()



def average_KNN(img_crop, model, emb_face, threshold=0.5):
	''' Nhận ảnh khuôn mặt đã được cắt và trả về nhận diện đối tượng và khoảng cách 
		từ ảnh tới các điểm dữ liệu trung bình
		Var -img_crop: ảnh khuôn mặt đã được cắt type = np.array
			-threshold: đặt ngưỡng nếu khoảng cách quá xa thì là class ko xác định, default = 0.5
	'''
	embeddings_list = [] # Chứa tất cả khoảng cách của face tới với toàn bộ dữ liệu
	names_list = [] # Chứa tên tất cả các đối tượng

	face = change_type(img_crop)

	output = model(face)

	for name in emb_face.keys():
		euclidean_distance = F.pairwise_distance(output, emb_face[name]) # Distance
		embeddings_list.append(euclidean_distance)
		names_list.append(name)

	min_list = min(embeddings_list)

	# Đặt ngưỡng nếu lớn hơn ngưỡng thì không xác định được danh tính
	if min_list > threshold:
		return 'No name', min_list.double()
	else:
		return names_list[embeddings_list.index(min_list)], min_list.double()