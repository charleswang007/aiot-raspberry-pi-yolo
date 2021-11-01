import cv2
import numpy as np 
import time
import requests
from datetime import date
from datetime import datetime

#today = date.today()
#print("Today's date:", today)

def load_yolo():
	net = cv2.dnn.readNet("/opt/mask_50/tiny_weights/yolov3-tiny_last.weights", "/opt/mask_50/cfg/yolov3-tiny.cfg")
#	net = cv2.dnn.readNetFromDarknet("/opt/mask_50/tiny_weights/yolov3-tiny_last.weights", "/opt/mask_50/cfg/yolov3-tiny.cfg")
#	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#	net = cv2.dnn.readNet("/opt/webcam_mask_flask/yolov3-tiny.weights", "/opt/webcam_mask_flask/yolov3-tiny-default.cfg")
#	net = cv2.dnn.readNet("/opt/webcam_mask_flask/yolov3.weights", "/opt/webcam_mask_flask/yolov3-default.cfg")
	classes = []
	with open("/opt/mask_50/cfg/obj.names", "r") as f:
#	with open("/opt/webcam_mask_flask/coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	count=0
	for i in layers_names:
		print (str(count)+"\t"+i)
		count=count+1
		
	print (layers_names)
	for i in net.getUnconnectedOutLayers():
		print (i)
		print (layers_names[i[0]-1])
	
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
#	output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def start_webcam():
	cap = cv2.VideoCapture(0)
	return cap

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#	net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
#	net.
#	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE) 
#	net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			
			if conf > 0.5:
#				print ("detect:",detect)
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
				
#				print ("scores:",scores, "class_id:",class_id, "conf:", conf, "center_x:", center_x, "center_y:", center_y, "w:",w,"h:",h,"x:",x,"y:",y)
	return boxes, confs, class_ids
			
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.0)
#	print ("indexes: ",indexes)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
#		print ("i:",i,"draw_labels:\t",i)
		if i>0:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			conf = confs[i]
			color = colors[i%3]

			if (y<10):
				y=y+40
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

			today = date.today()
			now = datetime.now()
			imgFileName = str(now).replace(":","_").replace(" ","_")+"_"+label+".jpg"
#			r = requests.post('http://localhost:5000/postLabel', json={"label":label,"x":x,"y":y,"w":w, "h":h, "img":imgFileName})
			
			print("now:", now, "conf:", conf, "label:", label, "x:",x, "y:",y, "w:",w, "h:",h, "img:", imgFileName)
#			cv2.imwrite(imgFileName, img)
	cv2.imshow("Image", img)

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()

if __name__ == '__main__':
	webcam_detect()
	cv2.destroyAllWindows()
