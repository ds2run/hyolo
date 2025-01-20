from ultralytics import YOLO
from PIL import Image
import cv2
import os

# print(os.getcwd())
model = YOLO("last.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="/datasets/coco128/images/train2017/000000000009.jpg", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("./ultralytics/datasets/coco128/images/train2017/000000000009.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("./ultralytics/datasets/coco128/images/train2017/000000000009.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
results = model.predict(source=[im1, im2])
# print(f"results: {results}")
success = model.export(format="onnx", opset=12)  # export the model to ONNX format