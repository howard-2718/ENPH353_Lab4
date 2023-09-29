#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import numpy as np

import cv2
import sys

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 0
		self._cam_fps = 20 # Was 10
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(int(1000 / self._cam_fps))

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)

		print("Loaded template image file: " + self.template_path)

	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		"""
        Create a Homography for image matching using SIFT algorithm.

        This function reads a frame from the camera, compares it with a template image using the SIFT algorithm,
        and updates the display based on the matching results.

        Returns:
            None

        """
		ret, frame = self._camera_device.read()
		
		# Read the template image and generate its keypoints and descriptors.
		temp_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		sift = cv2.SIFT_create()

		# Detect keypoints and compute descriptors for the template image.
		kp_temp, desc_temp = sift.detectAndCompute(temp_img, None)

		# Match the features between the template image and the webcam input.
		index_params = dict(algorithm=0, trees=5)
		search_params = dict()

		# Set up FLANN-based matcher
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		# Detect keypoints and compute descriptors for the webcam image.
		web_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		kp_web, desc_web = sift.detectAndCompute(web_img, None)

		matches = flann.knnMatch(desc_temp, desc_web, k=2)

		# Find the good points in the feature matching.
		good_points = []

		for m, n in matches:
			if m.distance < 0.6 * n.distance:
				good_points.append(m)

		# Create the 'comparison' image so that it can be shown on the UI.
		img_comp = cv2.drawMatches(temp_img, kp_temp, web_img, kp_web, good_points, web_img)

		# If we have more than 7 good matches, generate a Homography.
		if len(good_points) > 7:
			temp_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
			web_pts = np.float32([kp_web[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
			
			# Generate the Homography and the appropriate mask.
			M, mask = cv2.findHomography(temp_pts, web_pts, cv2.RANSAC, 5.0)
			matchesMask = mask.ravel().tolist()

			# Apply perspective transform to the generated Homography.
			h,w = temp_img.shape
			pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

			# Draw the image to the webcam feed.
			dst = cv2.perspectiveTransform(pts, M)
			frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

			pixmap = self.convert_cv_to_pixmap(frame)
			self.live_image_label.setPixmap(pixmap)
		else:
			# The homography was not generated, so show the 'comparison' image instead.
			pixmap = self.convert_cv_to_pixmap(img_comp)
			self.live_image_label.setPixmap(pixmap)
		
	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())
