from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2
import os
import urllib.request
import matplotlib.pyplot as plt

# Path to the font file
fontpath = "./arial.ttf"
backup_fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Download Arial Unicode font if not already present; if it fails, use DejaVuSans as a backup
try:
    if not os.path.exists(fontpath):
        url = "https://github.com/kormosi/arial-unicode-ms/raw/master/ARIALUNI.TTF"
        urllib.request.urlretrieve(url, fontpath)
        font = ImageFont.truetype(fontpath, 32)
    else:
        font = ImageFont.truetype(fontpath, 32)
except Exception as e:
    print("Unable to download font from URL or unsupported Unicode. Using DejaVuSans font instead.", e)
    font = ImageFont.truetype(backup_fontpath, 32)

# Load and process the image
img = cv2.imread('image-gg-1.jpg')
img = cv2.resize(img, (800, 600))  # Resize the image

# Set the text color as green
b, g, r, a = 0, 255, 0, 0

# Convert image to grayscale and apply blur
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

# Detect edges for contour detection
edged = cv2.Canny(blurred, 10, 200)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# Loop through contours to find the number plate shape (approx. 4-sided)
for c in contours:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approximation) == 4:
        number_plate_shape = approximation
        break

# Extract the area likely to contain the number plate text
(x, y, w, h) = cv2.boundingRect(number_plate_shape)
number_plate = grayscale[y:y + h, x:x + w]

# Use OCR to detect text on the number plate
reader = Reader(['en'])
detection = reader.readtext(number_plate)

# Set the detected text or a default message if no text is detected
if len(detection) == 0:
    text = "Number plate not found"
else:
    text = "Plate Number: " + detection[0][1]

# Convert the image to PIL format for text overlay
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((150, 500), text, font=font, fill=(b, g, r, a))  # Add text overlay
img = np.array(img_pil)

# Save the output image
output_path = "output_image-gg-1.jpg"
cv2.imwrite(output_path, img)
print(f"Image successfully saved at {output_path}")

# Display the image using matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')  # Hide axis
plt.show()
