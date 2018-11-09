from PIL import Image
from resizeimage import resizeimage
import os

directory = '/Users/research/Desktop/Skin Cancer Image Dataset'
for filename in os.listdir(directory):
	if filename.endswith(".jpg"):
		imgName = os.path.join(directory, filename)
		fd_img = open(imgName, 'r')
		img = Image.open(fd_img)
		img = resizeimage.resize_cover(img, [200, 200])
		img.save('resized -' + filename, img.format)
		img.save('/Users/research/Desktop/dataset/' + filename, img.format)
		fd_img.close()
		continue
	else:
		continue