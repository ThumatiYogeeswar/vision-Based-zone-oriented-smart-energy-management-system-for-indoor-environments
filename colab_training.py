!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="utFcLnRHvbZUS7XMzTIF")
project = rf.workspace("test-g2sru").project("coco-dataset-only-person")
version = project.version(1)
dataset = version.download("yolov8")

!pip install ultralytics

from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='/content/COCO-Dataset-Only-Person-1/data.yaml', epochs=15)

from google.colab import files

# Path to the best.pt file generated during training
file_path = '/content/runs/detect/train/weights/best.pt'

# Download the file
try:
    files.download(file_path)
    print(f"'{file_path}' downloaded successfully.")
except Exception as e:
    print(f"Error downloading '{file_path}': {e}")

from ultralytics import YOLO

# Load the best trained model (assuming 'best.pt' was saved from previous training)
model = YOLO('/content/runs/detect/train2/weights/best.pt') # Note: using 'train2' as per previous training output

# Run validation on the model to get the metrics
metrics = model.val()

# Print the requested metrics
print(f"Precision (metrics/precision(B)): {metrics.box.p}")
print(f"Recall (metrics/recall(B)): {metrics.box.r}")
print(f"mAP50 (metrics/mAP50(B)): {metrics.box.map50}")
print(f"mAP50-95 (metrics/mAP50-95(B)): {metrics.box.map}")

```python
# predict.py

from ultralytics import YOLO

# Load your trained model
# Make sure 'best.pt' is in the same directory as this script,
# or provide the full path to where you saved it.
model = YOLO('best.pt')

# Perform inference on an image
# Replace 'path/to/your/image.jpg' with the actual path to an image you want to test
results = model.predict(source='path/to/your/image.jpg', save=True, conf=0.5)

# Show results (optional)
for r in results:
    print(r.boxes)  # Print detection boxes
    print(r.probs)  # Print classification probabilities (if classification model)
    print(r.keypoints) # Print keypoints (if pose estimation model)

print("Inference complete! Check the 'runs/detect' folder for results.")
```

**To run this script:**

1.  Save the code above as `predict.py` in your VS Code project folder.
2.  Place an image you want to test (e.g., `test_image.jpg`) in the same folder, or update `source` with the correct path.
3.  Open the integrated terminal in VS Code.
4.  Activate your virtual environment (if you created one).
5.  Run the script:
    ```bash
    python predict.py
    ```

The results, including the image with bounding boxes, will be saved in a `runs/detect/predict` folder within your project directory.
