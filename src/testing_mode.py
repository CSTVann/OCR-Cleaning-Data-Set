import cv2
import easyocr
import matplotlib.pyplot as plt

def draw_bounding_boxes(image, detections, threshold=0.25):
    for bbox, text, score in detections:
        if score > threshold:
            # Increase the font scale to make the text bigger
            font_scale = 2.5  # Change the font scale here

            # Draw the bounding box and text on the image
            cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (255, 0, 0), 2)

# Set the relative path to the image file
image_path = "image/TestingWCT_Scope.png"

# Load the image
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Error loading the image. Please check the file path.")

# Perform OCR using EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
text_detections = reader.readtext(img)

# Print detected text to the terminal
counted_text = 0
for bbox, text, score in text_detections:
    if score > 0.25:
        counted_text += len(text.split())
print(f'Number of words: {counted_text-1}')

# Print detected text to the terminal
for bbox, text, score in text_detections:
    if score > 0.25:
        print(f'Text: {text}')

# Draw bounding boxes on the image
draw_bounding_boxes(img, text_detections, threshold=0.25)

# Convert BGR image to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

# Display the image with bounding boxes
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
