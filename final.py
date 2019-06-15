

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2



MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def initialize_caffe_models():

	age_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_age.prototxt',
		'data/age_net.caffemodel')

	gender_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_gender.prototxt',
		'data/gender_net.caffemodel')

	return(age_net, gender_net)




def read_from_camera(age_net, gender_net):
	font = cv2.FONT_HERSHEY_SIMPLEX


	# start the file video stream thread and allow the buffer to
	# start to fill
	print("[INFO] starting video file thread...")
	url='http://192.168.43.1:8080/video'
	fvs = FileVideoStream(url).start()
	time.sleep(1.0)

	# start the FPS timer
	fps = FPS().start()

	# loop over frames from the video file stream
	while fvs.more():
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale (while still retaining 3
		# channels)
		face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface_improved.xml')
		frame = fvs.read()
		frame = imutils.resize(frame, width=450)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


		faces = face_cascade.detectMultiScale(frame, 1.1, 5)
		frame = np.dstack([frame, frame, frame])


		if(len(faces)>0):
			print("Found {} faces".format(str(len(faces))))
			for (x, y, w, h )in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
				face_img = frame[y:y+h, h:h+w].copy()
				blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

			#Predict Gender
			gender_net.setInput(blob)
			gender_preds = gender_net.forward()
			gender = gender_list[gender_preds[0].argmax()]
			print("Gender : " + gender)

			#Predict Age
			age_net.setInput(blob)
			age_preds = age_net.forward()
			age = age_list[age_preds[0].argmax()]
			print("Age Range: " + age)

			overlay_text = "%s %s" % (gender, age)
			cv2.putText(frame, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)



















		# display the size of the queue on the frame
		cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		# show the frame and update the FPS counter
		cv2.imshow("Frame", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	fvs.stop()




if __name__ == "__main__":
	age_net, gender_net = initialize_caffe_models()

	read_from_camera(age_net, gender_net)
