import os
import detect_image

ROOT = os.getcwd() # Đường dẫn của file


def data_loader() -> dict:
	''' Tạo biến kiểu dữ liệu dict để chứa tên người, label và tên ảnh từng dữ liệu của người đó
	 Return data = {names_person : [label , [name_image1, name_image2, ...]]}
	'''
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



def save_crop_face(path=ROOT+'/runs/crops') -> None:
	''' Xóa thư mục ảnh và crop từng khuôn mặt trên ảnh trong Person_data
		và lưu trữ tại runs/crops/
	'''
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


if __name__ == '__main__':
	save_crop_face()