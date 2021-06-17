# USAGE
# python recognition.py 


# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
from imutils import face_utils

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("recognizer.pickle", "rb").read())
le = pickle.loads(open("labels.pickle", "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(src=0).start()

while True:

		# grab the next frame from the stream, resize it and flip it
		# horizontally
		frame = vs.read()
		h, w, _ = frame.shape
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (640, 480))
		img_mean = np.array([127, 127, 127])
		img = (img - img_mean) / 128
		img = np.transpose(img, [2, 0, 1])
		img = np.expand_dims(img, axis=0)
		rgb= img.astype(np.float32)


		# convert the frame from RGB (OpenCV ordering) to dlib 
		# ordering (RGB)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb)

		# loop over the face detections
		for (top, right, bottom, left) in boxes:
			# draw the face detections on the frame
			cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)

		# check if atleast one face has been detected	
		if len(boxes) > 0:
			# compute the facial embedding for the face
			encodings = face_recognition.face_encodings(rgb, boxes)
			preds = recognizer.predict_proba(encodings)[0]
			j = np.argmax(preds)
			name = le.classes_[j]

			label = "hello this is {}".format(name)
			cv2.putText(frame, label, (5, 175),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		# show the frame
		cv2.imshow("face recognition", frame)
		key = cv2.waitKey(1) & 0xFF
		
		# check if the `q` key was pressed
		if key == ord("q"):
				break

vs.stop()
