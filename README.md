# Face Recognition App with LBPH and Haar Cascades

A complete Streamlit-based face recognition application using Local Binary Patterns Histograms (LBPH) and Haar Cascades for face detection and recognition.

## Features

- **Face Detection**: Uses Haar Cascades for robust face detection
- **Face Recognition**: Implements LBPH algorithm for accurate face recognition
- **Training Mode**: Upload multiple images to train the model on different people
- **Recognition Mode**: Recognize trained faces in new images
- **Interactive UI**: User-friendly Streamlit interface
- **Confidence Threshold**: Adjustable threshold for recognition sensitivity
- **Visual Results**: Displays bounding boxes and confidence scores

## Installation

### Requirements
- Python 3.7+
- OpenCV with contrib modules
- Streamlit
- Pillow
- NumPy

### Install Dependencies

```bash
pip install streamlit opencv-contrib-python pillow numpy
```

## How to Run

```bash
streamlit run face_recognition_app.py
```

The app will open in your default browser at `http://localhost:8501`

## How to Use

### 1. Train the Model

1. Select **"Train Model"** from the sidebar
2. Set the number of people you want to train
3. For each person:
   - Enter their name
   - Upload 3-5 clear images of their face
4. Click **"Train Model"** button
5. Wait for training to complete (you'll see a success message with balloon animation)

### 2. Recognize Faces

1. Select **"Recognize Faces"** from the sidebar
2. Adjust the confidence threshold if needed (lower = stricter matching)
3. Upload an image to test
4. View the results:
   - Recognized faces are marked with green boxes
   - Unknown faces are marked with red boxes
   - Confidence scores are displayed for each detection

## Technical Details

### Haar Cascades
- Pre-trained classifier for face detection
- Fast and efficient for frontal face detection
- Uses cascade of simple features

### LBPH (Local Binary Patterns Histograms)
- **Radius**: 1 (distance to neighboring pixels)
- **Neighbors**: 8 (number of sample points)
- **Grid**: 8x8 (divides face into regions)
- Robust to illumination changes
- Works well with grayscale images

### Face Processing Pipeline
1. Convert image to grayscale
2. Detect faces using Haar Cascade
3. Extract and resize face regions to 200x200
4. Extract LBP features
5. Compare with trained model
6. Return label and confidence score

## Tips for Best Results

### Training Images
- Use 3-5 images per person minimum
- Include different angles and expressions
- Ensure good lighting (avoid shadows)
- Use clear, high-resolution images
- Make sure faces are fully visible (no obstruction)

### Recognition
- Use images with similar lighting to training data
- Ensure faces are clearly visible
- Front-facing photos work best
- Adjust confidence threshold based on your needs:
  - Lower (40-60): Stricter matching, fewer false positives
  - Medium (60-80): Balanced
  - Higher (80-100): More lenient, may include uncertain matches

## Confidence Score Interpretation

- **0-40**: Very confident match
- **40-60**: Confident match
- **60-80**: Moderate match (default threshold: 70)
- **80-100**: Uncertain match (usually marked as "Unknown")
- **>100**: No match

Lower confidence values indicate better matches!

## Limitations

- Works best with frontal face images
- Lighting conditions can affect accuracy
- Requires multiple training images per person
- May struggle with:
  - Extreme facial expressions
  - Partial face occlusion
  - Poor lighting
  - Low image quality

## Project Structure

```
face_recognition_app.py       # Main application file
README.md                      # This file
```

## Troubleshooting

### No faces detected during training
- Ensure images contain clear, visible faces
- Check image quality and lighting
- Try images with more frontal views

### Poor recognition accuracy
- Add more training images per person (5-10 recommended)
- Use images with consistent lighting
- Lower the confidence threshold
- Ensure training and test images have similar conditions

### App won't start
- Verify all dependencies are installed
- Check Python version (3.7+ required)
- Ensure opencv-contrib-python is installed (not just opencv-python)

## Future Enhancements

Potential improvements for this application:
- Save/load trained models
- Webcam support for live recognition
- Database integration for storing face data
- Multiple face recognition algorithms
- Performance metrics and accuracy testing
- Batch image processing

## License

This project is open source and available for educational purposes.

## Credits

- OpenCV for face detection and recognition algorithms
- Streamlit for the web interface
- Haar Cascades for face detection
