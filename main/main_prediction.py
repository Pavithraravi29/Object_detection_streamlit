from ultralytics import YOLO
from PIL import Image
import cv2

# model = YOLO('yolov8n.pt')
model = YOLO("C:\\Users\\SDC-03\\Desktop\\image_detection\\main\\runs\\detect\\train28\\weights\\best.pt")

# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
IMAGE_PATH = "C:\\Users\\SDC-03\\Desktop\\image_detection\\main\\datasets\\my_dataset\\images\\test\\5e0264f9-image_31.png"
# IMAGE_PATH = "D:\\image_detection\\main\\datasets\\my_dataset\\images\\test\\test_222.jpg"

im1 = Image.open(IMAGE_PATH)

results = model.predict(source=im1, save=True)  # save plotted images

print("===================")
print(results)
print("===================")
# from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])
