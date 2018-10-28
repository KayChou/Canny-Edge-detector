import numpy as np
import cv2
from matplotlib import pyplot as plt


# ===========================================================
# plot the image's histogram
# ===========================================================
def img_hist(img):
	hist = np.zeros(256)
	shape = img.shape
	for i in range(shape[0]):
		for j in range(shape[1]):
			hist[img[i][j]] = hist[img[i][j]] + 1
	return hist


# ===========================================================
# Grayscale histogram specification
# ===========================================================
def specification(img):
	hist = img_hist(img)
	shape = img.shape
	p_k = hist/(shape[0]*shape[1])

	s_k = np.zeros(256)
	s = 0
	for i in range(256):
		s = s + p_k[i]
		s_k[i] = s

	Look_up_Table = 256*s_k
	img_spec = np.zeros_like(img)
	for i in range(shape[0]):
		for j in range(shape[1]):
			img_spec[i][j] = Look_up_Table[img[i][j]]

	return img_spec


def hist_show(img1, img2):
	plt.subplot(2, 1, 1)
	plt.bar(range(256), img1)
	plt.title("Original Grayscale histogram")
	plt.subplot(2, 1, 2)
	plt.bar(range(256), img2)
	plt.title("Grayscale histogram Specification")
	plt.show()


def img_show(img1, img2):
	shape = img1.shape
	dim1 = 600
	dim2 = int(dim1*shape[0]/shape[1])

	im1 = cv2.resize(img1, (dim1, dim2))
	im2 = cv2.resize(img2, (dim1, dim2))
	hmerge = np.hstack((im1, im2))
	cv2.imshow("Image", hmerge)
	# vmerge = np.vstack((im1, im2))
	cv2.waitKey()
	cv2.destroyAllWindows()


def main():
	img = cv2.imread("lena.jpg")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print ("Source Image Shape: ", img.shape, "\nSource Image Type: ", type(img))
	hist1 = img_hist(img)
	img_spec = specification(img)
	hist2 = img_hist(img_spec)

	img_show(img, img_spec)
	hist_show(hist1, hist2)


if __name__ == '__main__':
	main()







'''
cv2.imshow("Image",img)
cv2.waitKey(0)

def image_hist(image):
    color = ('b', 'g', 'r')
    for i , color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color)
        plt.xlim([0, 256])
    plt.show()


def img_hist_cv2(img):
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	plt.plot(hist)
	plt.show()


'''
