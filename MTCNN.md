## Program Description: How It Works
This program aims to detect whether a person in a video appears "drunk" or "sober" based on face detection and red color detection on their faces. The face detection is performed using the MTCNN (Multi-Task Cascaded Convolutional Neural Network) algorithm, while the red color detection is implemented using color conversion to the HSV color space and thresholding.

### Libraries Used
- **OpenCV (`cv2`)**: Used for image processing tasks such as reading frames from a webcam, color conversion, and drawing rectangles and text on images.
- **NumPy (`np`)**: Utilized for numerical operations and array manipulation.
- **MTCNN (`mtcnn`)**: A face detection library used to detect faces in images.
- **TensorFlow (`tf`)**: Deep learning framework for loading and using pre-trained models.

## MTCNN is a deep learning-based face detection algorithm that detects faces in an image using three stages: 
1. **Proposal Network (P-Net)**: Generates candidate face regions.
2. **Refinement Network (R-Net)**: Filters out non-face regions and refines the bounding boxes.
3. **Output Network (O-Net)**: Refines the results further and detects facial landmarks.

### Model Training
- The model used for eye detection is trained separately using TensorFlow.
- Details of the training process, including data collection, preprocessing, model architecture, compilation, and training, are not included in this script.
- The trained model (`facial_drunk.keras`) is loaded using TensorFlow's `load_model` function.

### Methodology
1. **Face Detection**:
   - Utilizes the MTCNN algorithm to detect faces in each frame of the video.

2. **Red Color Detection**:
   - After faces are detected, the face area is taken as the Region of Interest (ROI).
   - The RGB color space is converted to the HSV color space for easier color detection.
   - The desired range of red color is specified in the HSV color space.
   - Thresholding is performed to obtain a mask based on the specified red color range.
   - Count the number of non-black pixels in the mask.
   - Calculate the ratio of the number of red pixels to the total pixels in the face area.

3. **Detection Decision**:
   - If the ratio of red pixels exceeds a certain threshold, the face is considered "drunk".
   - Otherwise, it is considered "sober".

4. **Visual Representation**:
   - Detected faces are surrounded by bounding boxes.
   - The label "Drunk" or "Sober" is displayed above the bounding box according to the detection decision.
   - The color of the bounding box and label is adjusted based on the detection decision.