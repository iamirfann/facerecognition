# USAGE
# python collect_faces.py --dataset face_dataset --name johndoe 

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
# give dataset path for saving the face images
# give ur name 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
	help="Enter the dataset path for saving the images")
ap.add_argument("-n", "--name", required=True, 
	help="Name of the student")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(src=0).start()

# initialize the number of face detections and the total number
# of images saved to disk 
faceCount = 0
total = 0

status = "detecting"

# create the directory to store the student's data
os.makedirs(os.path.join(args["dataset"], 
	args["name"]), exist_ok=True)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it (so
	# face detection will run faster), flip it horizontally, and
	# finally clone the frame (just in case we want to write the
	# frame to disk later)
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	frame = cv2.flip(frame, 1)
	orig = frame.copy()
		
	# convert the frame from from RGB (OpenCV ordering) to dlib
	# ordering (RGB) and detect the (x, y)-coordinates of the
	# bounding boxes corresponding to each face in the input image
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	boxes = face_recognition.face_locations(rgb)
		
	# loop over the face detections
	for (top, right, bottom, left) in boxes:
		# draw the face detections on the frame
		cv2.rectangle(frame, (left, top), (right, bottom), 
			(0, 255, 0), 2)

		# this 30 represents 30 face detection to be done
		# if you need more u can change
		if faceCount < 30:
			# increment the detected face count and set the
			# status as detecting face
			faceCount += 1
			status = "detecting"
			continue

		# save the frame to correct path and increment the total 
		# number of images saved
		p = os.path.join(args["dataset"], 
			args["name"], "{}.png".format(str(total).zfill(5)))
		cv2.imwrite(p, orig[top:bottom, left:right])
		total += 1

		# set the status as saving frame 
		status = "saving"

	# draw the status on to the frame
	cv2.putText(frame, "Status: {}".format(status), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)

	#  this 30 represent face image to be stored in the dataset
	if total == 30:
		break


# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()


