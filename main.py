import detect_image, detect_video
import os
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np



class YOLOV5():
	''' YOLOV5 load file detect_image từ file YoloV5 để tách và phân loại đối tượng
		Ngoài ra crop từng object face trong ảnh để qua bước nhận diện 
	'''
	def __init__(self, source, path):
		self.source = source # Đường dẫn của hình ảnh dùng để test
		self.path = path # Đường dẫn hiện tại của file main.py


	def take_object(self):
		# Lấy các thông tin của từng object trong ảnh là center x,y,x,y và id class từ model YOLOV5
		output = detect_video.run(
			weights=self.path + '/Model_Parameter/YOLOv5/weight_train.pt',
			source= 0, #self.path + '/Images_test/', # file/dir/URL/glob, 0 for webcam
			data=self.path + '/data/facemask.yaml',
			imgsz=(640, 640),  # inference size (height, width)
			conf_thres=0.7,  # confidence threshold
			iou_thres=0.8,  # NMS IOU threshold
			max_det=1000,
			save_crop=False,  # save cropped prediction boxes
			view_img=False,  # maximum detections per image
			project=self.path + '/Detect',  # save results to project/name
			name='Face_Mask',  # save results to project/name
			)

		return output
		

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



class FaceNet():
	''' FaceNet dùng để train và nhận diện từng khuôn mặt dược tách ra bởi YoloV5
	'''
	def __init__(self, path):
		self.resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=3).eval() # initializing resnet for face img to embeding conversion
		saved_data = torch.load(path) # loading data.pt file
		self.embedding_list = saved_data[0] # getting embedding data
		self.name_list = saved_data[1] # getting list of names


	def parameter_person(self):
		mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

		dataset = datasets.ImageFolder('Person') # Person folder path 
		idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

		def collate_fn(x):
			return x[0]

		loader = DataLoader(dataset, collate_fn=collate_fn)

		face_list = [] # list of cropped faces from Person folder
		name_list = [] # list of names corrospoing to cropped photos
		embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

		for img, idx in loader:
			face, prob = mtcnn(img, return_prob=True) 
			if face is not None and (prob > 0.70): # if face detected and porbability > 90%
				emb = self.resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
				embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
				name_list.append(idx_to_class[idx]) # names are stored in a list

		data = [embedding_list, name_list]
		torch.save(data, 'Model_Parameter/FaceNet/Face_data01.pt') # saving Face_data.pt file


	def face_match(self, face): # face= image of object face, type là np.array 
		# Tìm khoảng cách nhỏ nhất với tập dữ liệu và trả về kết quả
		face = torch.from_numpy(face) #Change to tensor
		face = face.unsqueeze(0) #Flatten
		face = face.permute(0, 3, 1, 2).type(torch.float32) #Change size and type data

		emb = self.resnet(face).detach() # detech is to make required gradient false

		dist_list = [] # list of matched distances, minimum distance is used to identify the person
		for idx, emb_db in enumerate(self.embedding_list):
			dist = torch.dist(emb, emb_db).item()
			dist_list.append(dist)

		dist_team = sorted(dist_list)
		d1, d2, d3 = dist_team[0], dist_team[1], dist_team[2] 

		#idx_min = dist_list.index(d)
		return ([self.name_list[dist_list.index(d1)], 
			self.name_list[dist_list.index(d2)],
			self.name_list[dist_list.index(d3)]],
			[d1,d2,d3])



def identify_images(path_dict, path_images_test, path_faceNet, view_result, save_image):
	''' Liên kết các model lại với nhau để ra kết quả sau cùng
		Hiển thị và lưu kết quả
	'''
	yolo = YOLOV5(
		path=path_dict, # Đường dẫn của file main.py
		source=path_images_test # Đường dẫn của hình ảnh dùng để test
		) 
	fNet = FaceNet(
		path=path_faceNet # Đường dẫn file model đã train sẵn để nhận diện khuôn mặt
		) 

	# Detect Object
	data = yolo.take_object() # Type data: dict
	paths = data.keys() # Load path của tất cả hình ảnh test
	colors = {
		'Not Mask': (0, 0, 255),
		'Correct': (0, 255, 0),
		'Incorrect': (0, 255, 255)} # Màu sắc của từng class

	# Xóa data trong file nơi lưu hình ảnh đã nhận diện
	if save_image:
		name = 0
		os.chdir(path_dict+'/Detect')
		arr = os.listdir()
		for char in arr:
			os.remove(char)
		os.chdir(path_dict)

	# Nhận diện với từng bức ảnh
	for path in paths:
		im = cv2.imread(path)
		im0 = im.copy()

		n = len(data[path]) # Đếm số đối tượng trên mỗi bức ảnh

		# Nhận diện từng đối tượng trên bức ảnh
		for i in range(n):
			img_crop = yolo.detect_box(data[path][i][1], im) # Ảnh khuôn mặt đã được cắt
			
			result = fNet.face_match(img_crop) # Nhận kết quả nhận diện khuôn mặt
			print('Face matched with: ',result[0], 'With the smallest distance: ',result[1])

			# Open CV để hiển thị kết quả trên ảnh
			if view_result or save_image:
				x1, y1, x2, y2 = data[path][i][1]
				w, h = int(x2-x1), int(y2-y1)
				x, y = int((x1+x2)/2 - w/2), int((y1+y2)/2 - h/2)

				if im.shape[0]>im.shape[1]: size=im.shape[1]/500
				else: size=im.shape[0]/500

				name_class = data[path][i][0]
				cv2.putText(im0, name_class, (x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=size, color=colors[name_class])
				cv2.putText(im0, result[0][0],(x,y+h+int(h*0.15)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=size, color=colors[name_class])
				cv2.rectangle(im0, (x,y), (x+w, y+h), colors[name_class], 5)

		# Demo kết quả
		if view_result:
			cv2.imshow('image', im0)
			cv2.waitKey(0)

		# Lưu lại kết quả
		if save_image:
			cv2.imwrite(path_dict+'/Detect/'+str(name)+'.jpg', im0)
			name += 1



if __name__ == '__main__':
	curr_dict = os.getcwd()
	identify_images(
		path_dict= curr_dict, # Đường dẫn của file main.py
		path_images_test=curr_dict + '/Images_test/', # Đường dẫn của hình ảnh dùng để test
		path_faceNet='Model_Parameter/FaceNet/Face_data01.pt', # Đường dẫn file model đã train sẵn để nhận diện khuôn mặt
		view_result=False, # Demo kết quả
		save_image=True # Lưu lại kết quả
		)

	# fNet = FaceNet(
	# 	path='Model_Parameter/FaceNet/Face_data.pt') # Đường dẫn file model đã train sẵn để nhận diện khuôn mặt

	# fNet.parameter_person()

	# yolo = YOLOV5(
	# 	path=curr_dict, # Đường dẫn của file main.py
	# 	source=curr_dict + '/Images_test/') # Đường dẫn của hình ảnh dùng để test

	# yolo.take_object()