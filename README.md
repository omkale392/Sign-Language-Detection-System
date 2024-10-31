# Sign Language Detection System

A real-time sign language detection system that uses deep learning and computer vision to recognize and interpret hand gestures from webcam feed. The system employs LSTM neural networks and MediaPipe for accurate hand tracking and gesture recognition.

## Features
- Real-time hand gesture recognition using webcam feed
- Deep learning model based on 3-layer LSTM architecture
- Hand landmark detection and tracking using MediaPipe
- Confidence score visualization for predictions
- Support for multiple gesture classes (currently A, B, C)
- Real-time visualization of detection probabilities

## Technologies Used
- Python 3.x
- TensorFlow/Keras for deep learning
- OpenCV for image processing and visualization
- MediaPipe for hand landmark detection
- NumPy for numerical computations
- scikit-learn for data preprocessing

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/SignLanguageDetectionSystem.git
cd SignLanguageDetectionSystem
```

2. Install required packages
```bash
pip install tensorflow opencv-python mediapipe numpy sklearn
```

## Project Structure
- `app.py`: Main application file for real-time detection
- `collectdata.py`: Script for collecting training data
- `data.py`: Data preprocessing and preparation
- `function.py`: Utility functions for detection and visualization
- `trainmodel.py`: Model training script
- `model.h5`: Trained model weights
- `model.json`: Model architecture

## Usage

1. Collect training data:
```bash
python collectdata.py
```
Make the appropriate hand sign and press the corresponding key to capture images.

2. Process the collected data:
```bash
python data.py
```

3. Train the model:
```bash
python trainmodel.py
```

4. Run the detection system:
```bash
python app.py
```

## System Workflow
1. The system captures video feed from the webcam
2. Detects hand landmarks using MediaPipe
3. Processes the landmarks through the LSTM model
4. Displays the recognized gesture with confidence score
5. Shows real-time visualization of detection probabilities

## Model Architecture
- Input Layer: 30 frames × 63 features (hand landmarks)
- LSTM Layer 1: 64 units with ReLU activation
- LSTM Layer 2: 128 units with ReLU activation
- LSTM Layer 3: 64 units with ReLU activation
- Dense Layers: 64 → 32 → output classes
- Output: Softmax activation for gesture classification

## Future Improvements
- Support for more gesture classes
- Enhanced visualization options
- Mobile device support
- Real-time text generation from gestures
- Support for two-handed gestures

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
