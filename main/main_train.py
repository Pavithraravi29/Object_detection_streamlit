from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
#results = model.train(data='coco128.yaml', epochs=3)
results = model.train(data='my_data.yaml', epochs=500)

# Evaluate the model's performance on the validation set

results = model.val()

# Perform object detection on an image using the model
results = model(r'C:\Users\SDC-03\Desktop\image_detection\main\datasets\my_dataset\images\test\fd7d0c8e-image_28.png')

print("======================")
print(results)
print("======================")
# Export the model to ONNX format
success = model.export(format='onnx')
