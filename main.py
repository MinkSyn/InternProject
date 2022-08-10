import os
from PIL import Image
import cv2
import numpy as np

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import detect_image, detect_video

ROOT = os.getcwd()



class YOLOV5():
	''' YOLOV5 load file detect_image từ file YoloV5 để tách và phân loại đối tượng
		Ngoài ra crop từng object face trong ảnh để qua bước nhận diện 
	'''
	def __init__(self, source):
		self.source = source # Đường dẫn của hình ảnh dùng để test

	def take_object(self):
		# Lấy các thông tin của từng object trong ảnh là center x,y,x,y và id class từ model YOLOV5
		if self.source != 0:
			output = detect_image.run(
				weights=ROOT + '/Model_Parameter/YOLOv5/weight_train.pt',
				source= ROOT + self.source, # file/dir/URL/glob, 0 for webcam
				data=ROOT + '/data/facemask.yaml',
				imgsz=(640, 640),  # inference size (height, width)
				conf_thres=0.25,  # confidence threshold
				iou_thres=0.45,  # NMS IOU threshold
				max_det=1000,
				save_crop=False,  # save cropped prediction boxes
				view_img=False,  # maximum detections per image
				project=ROOT + '/Detect',  # save results to project/name
				name='Face_Mask',  # save results to project/name
				)

			return output

		else:
			detect_video.run(
				weights=self.path + '/Model_Parameter/YOLOv5/weight_train.pt',
				source= 0, # file/dir/URL/glob, 0 for webcam
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



class faceNet():
	def __init__(self):
		self.names_person = os.listdir(ROOT+ '/Person_data/') # List chứa các tên người nhận diện		
		self.transformation = transforms.Compose([transforms.Resize((100,100)),
									transforms.ToTensor()
									]) # Định dạng ảnh trc khi qua siamese neural network

	def face_match(self, img_crop, threshold=0.5): # face= image of object face, type là np.array 
		# Tìm khoảng cách nhỏ nhất với tập dữ liệu và trả về kết quả
		embeddings_list = [] # Chứa tất cả khoảng cách của face tới với toàn bộ dữ liệu
		names_list = [] # Chứa tên tất cả các đối tượng
		face1 = self.change_type_data(img_crop, PIL=False)

		for name in self.names_person:
			all_name_images = os.listdir(ROOT+ '/Person_data/'+ name +'/')

			for name_img in all_name_images:
				path = ROOT+'/runs/crops/'+name+'/'+name_img
				face2 = self.change_type_data(path)

				output1, output2 = net(face1, face2) # Embedding từ siamese neural network
				euclidean_distance = F.pairwise_distance(output1, output2)
				embeddings_list.append(euclidean_distance)
				names_list.append(name)

		min_dist = min(embeddings_list)
		if min_dist > threshold:
			return 'No_name', -1
		else:
			return names_list[embeddings_list.index(min_dist)], min_dist.double()


	def change_type_data(self, path, PIL=True):
		''' Chuyển đổi dạng dữ liệu
			Var: path đường dẫn hoặc ảnh có type = np.array
				 PIL xác định ảnh là dạng PIL hay không
		'''
		if PIL:
			img = Image.open(path) 
			img = img.convert("L")
		else:
			img_convert = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY) # Chuyển về ảnh xám
			img = Image.fromarray(img_convert) # Chuyển về định dang PIL

		img_trans = self.transformation(img) # Định dạng lại ảnh
		img_numpy3 = img_trans.numpy()
		img_numpy4 = np.array([img_numpy3]) # Tăng số chiều cho ảnh

		return torch.from_numpy(img_numpy4)

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

fNet = faceNet()
yolo = YOLOV5(
	source='/Images_test/' # Đường dẫn của hình ảnh dùng để test
	)
net = torch.load(ROOT + '/Model_Parameter/FaceNet/seamese_net.pth')



def identify_images(view_result, save_image):
	''' Liên kết các model lại với nhau để ra kết quả sau cùng
		Hiển thị và lưu kết quả
	'''
	
	data = yolo.take_object() # # Detect Object, type data return: dict
	paths = data.keys() # Load path của tất cả hình ảnh test
	colors = {
		'Not Mask': (0, 0, 255),
		'Correct': (0, 255, 0),
		'Incorrect': (0, 255, 255)} # Màu sắc của class

	# Xóa data trong file nơi lưu hình ảnh đã nhận diện
	if save_image:
		name = 0
		os.chdir(ROOT + '/Detect')
		arr = os.listdir()
		for char in arr:
			if char[-4:] == '.jpg':
				os.remove(char)
		os.chdir(ROOT)

	# Nhận diện với từng bức ảnh
	for path in paths:
		im = cv2.imread(path)
		im0 = im.copy()

		n = len(data[path]) # Đếm số đối tượng trên mỗi bức ảnh

		# Nhận diện từng đối tượng trên bức ảnh
		for i in range(n):
			img_crop = yolo.detect_box(data[path][i][1], im) # Ảnh khuôn mặt đã được cắt	
			result, distance = fNet.face_match(img_crop) # Nhận kết quả nhận diện khuôn mặt
			print('Object is {}, distance is {}'.format(result, distance))

			# Open CV để hiển thị kết quả trên ảnh
			if view_result or save_image:
				x1, y1, x2, y2 = data[path][i][1]
				w, h = int(x2-x1), int(y2-y1)
				x, y = int((x1+x2)/2 - w/2), int((y1+y2)/2 - h/2)

				if im.shape[0]>im.shape[1]: size=im.shape[1]/500
				else: size=im.shape[0]/500

				name_class = data[path][i][0]
				if result == 'No_Name':
					cv2.putText(im0, str(result), (x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=size, color=colors[name_class])
				else: 
					cv2.putText(im0, str(result)+', '+str(distance)[:6], (x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=size, color=colors[name_class])
				cv2.putText(im0, name_class, (x,y+h+int(h*0.1)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=size, color=colors[name_class])		
				cv2.rectangle(im0, (x,y), (x+w, y+h), colors[name_class], 5)

		# Demo kết quả
		if view_result:
			cv2.imshow('image', im0)
			cv2.waitKey(0)

		# Lưu lại kết quả
		if save_image:
			cv2.imwrite(ROOT+'/Detect/'+str(name)+'.jpg', im0)
			name += 1



if __name__ == '__main__':
	identify_images(
		view_result=False, # Demo kết quả
		save_image=True # Lưu lại kết quả
		)