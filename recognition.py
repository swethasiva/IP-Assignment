import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[classids[i]]]
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:10f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > tconf:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        net.setInput(blob)

        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs

FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')


	FLAGS = parser.parse_args()

	labels = open(FLAGS.labels).read().strip().split('\n')

	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	if FLAGS.image_path is None and FLAGS.video_path is None:
	    print ('Neither path to an image or path to video provided')
	    print ('Starting Inference on Webcam')
	    
	if FLAGS.image_path:
		try:
			img = cv.imread(FLAGS.image_path)
			height, width = img.shape[:2]
		except:
			raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			img, _, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
			cv.imwrite('output.png',img)
			show_image(img)

	elif FLAGS.video_path:
		try:
			vid = cv.VideoCapture(FLAGS.video_path)
			height, width = None, None
			writer = None
		except:
			raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			while True:
				grabbed, frame = vid.read()
				if not grabbed:
					break

				if width is None or height is None:
					height, width = frame.shape[:2]

				frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

				if writer is None:
					fourcc = cv.VideoWriter_fourcc(*"MJPG")
					writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
						            (frame.shape[1], frame.shape[0]), True)


				writer.write(frame)

			writer.release()
			vid.release()


