import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set the environment variable to use offscreen rendering
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Load YOLO model and classes
net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

# Input and output directories
input_directory = "test"
output_directory = "outy_images"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Lists for evaluation
all_labels = []
all_predictions = []

# Process each image in the input directory
for image_name in os.listdir(input_directory):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_directory, image_name)

        # Read the image
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Prepare the image for inference
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        # Process the detections
        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Load ground truth labels from the corresponding .txt file
        label_file_path = os.path.join(input_directory, image_name.replace('.jpg', '.txt'))
        with open(label_file_path, 'r') as label_file:
            ground_truth_labels = [line.strip().split() for line in label_file]

        # Compare ground truth and predictions
        for ground_truth_label in ground_truth_labels:
            class_idx, x_center, y_center, box_width, box_height = map(float, ground_truth_label)

            # YOLO format uses normalized coordinates, so multiply by width and height
            x, y, w, h = map(int, [x_center * width, y_center * height, box_width * width, box_height * height])

            # Save ground truth
            all_labels.append(int(class_idx))

            # Draw bounding boxes on the image for ground truth
            color_gt = (0, 255, 0)  # Green for ground truth
            cv2.rectangle(img, (x, y), (x + w, y + h), color_gt, 2)

        # Draw predicted boxes after NMS
        for i in indices:
            i = i[0]  # Extract the index from the list
            x_pred, y_pred, w_pred, h_pred = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = round(confidences[i], 2)
            all_predictions.append(class_ids[i])

            # Draw bounding boxes on the image for predictions
            color_pred = (0, 0, 255)  # Red for predictions
            cv2.rectangle(img, (x_pred, y_pred), (x_pred + w_pred, y_pred + h_pred), color_pred, 2)

        # Save the annotated image
        annotated_image_path = os.path.join(output_directory, f"annotated_{image_name}")
        cv2.imwrite(annotated_image_path, img)

# Evaluate the model
print("Model Evaluation:")
print(classification_report(all_labels, all_predictions, target_names=classes))
cm = confusion_matrix(all_labels, all_predictions, labels=range(len(classes)))

# Calculate additional accuracy metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision_macro = precision_score(all_labels, all_predictions, average='macro')
recall_macro = recall_score(all_labels, all_predictions, average='macro')
f1_macro = f1_score(all_labels, all_predictions, average='macro')

# Save additional accuracy metrics to a text file
with open('accuracy_metricss.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy:.4f}\n')
    file.write(f'Precision (Macro): {precision_macro:.4f}\n')
    file.write(f'Recall (Macro): {recall_macro:.4f}\n')
    file.write(f'F1 Score (Macro): {f1_macro:.4f}\n')

# Plot confusion matrix and save the plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrixt.png')
plt.close()  # Close the plot without displaying it

