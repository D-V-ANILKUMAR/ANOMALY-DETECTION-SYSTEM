import requests
import time
import subprocess
import os
import cv2
import numpy as np

# Create a dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[25:75, 25:75] = [255, 0, 0] # Anomaly
cv2.imwrite('dummy.png', img)

print("Starting server...")
proc = subprocess.Popen(['python', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3)

print("Sending request...")
try:
    with open('dummy.png', 'rb') as f:
        response = requests.post('http://127.0.0.1:5000/api/detect', files={'image': f}, data={'image_type': 'rgb', 'method': 'mahalanobis'})
    print(response.status_code)
    print(response.text[:200])
except Exception as e:
    print("Error:", e)

proc.terminate()
