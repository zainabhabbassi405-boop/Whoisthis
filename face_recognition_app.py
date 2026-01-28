import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pickle

# Set page configuration
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="👤",
    layout="wide"
)

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'label_mapping' not in st.session_state:
    st.session_state.label_mapping = {}

# Load Haar Cascade
@st.cache_resource
def load_haar_cascade():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cascade

def detect_faces(image, cascade):
    """Detect faces in an image using Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, gray

def prepare_training_data(images, labels):
    """Prepare training data by detecting faces and extracting face regions"""
    cascade = load_haar_cascade()
    faces = []
    face_labels = []
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        detected_faces, gray = detect_faces(image, cascade)
        
        for (x, y, w, h) in detected_faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            faces.append(face_roi)
            face_labels.append(label)
    
    return faces, face_labels

def train_lbph_model(faces, labels):
    """Train LBPH face recognizer"""
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    recognizer.train(faces, np.array(labels))
    return recognizer

def recognize_face(image, recognizer, cascade, label_mapping, confidence_threshold=70):
    """Recognize faces in an image"""
    detected_faces, gray = detect_faces(image, cascade)
    results = []
    
    for (x, y, w, h) in detected_faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        
        label, confidence = recognizer.predict(face_roi)
        
        if confidence < confidence_threshold:
            name = label_mapping.get(label, f"Unknown (ID: {label})")
            results.append({
                'bbox': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })
        else:
            results.append({
                'bbox': (x, y, w, h),
                'name': 'Unknown',
                'confidence': confidence
            })
    
    return results

def draw_results(image, results):
    """Draw bounding boxes and labels on image"""
    output_image = image.copy()
    
    for result in results:
        x, y, w, h = result['bbox']
        name = result['name']
        confidence = result['confidence']
        
        # Draw rectangle
        color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"{name} ({confidence:.1f})"
        cv2.putText(output_image, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return output_image

# Streamlit App
st.title("👤 Face Recognition App")
st.markdown("**Using LBPH (Local Binary Patterns Histograms) and Haar Cascades**")

# Sidebar
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Select Mode:", ["Train Model", "Recognize Faces"])

cascade = load_haar_cascade()

# Train Model Mode
if mode == "Train Model":
    st.header("🎓 Train Face Recognition Model")
    st.markdown("Upload multiple images for each person to train the model.")
    
    num_people = st.number_input("Number of people to train:", min_value=1, max_value=20, value=2)
    
    training_data = {}
    
    for i in range(num_people):
        st.subheader(f"Person {i+1}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            person_name = st.text_input(f"Name:", key=f"name_{i}", value=f"Person {i+1}")
        
        with col2:
            uploaded_files = st.file_uploader(
                f"Upload images for {person_name}:", 
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key=f"upload_{i}"
            )
        
        if uploaded_files:
            training_data[i] = {
                'name': person_name,
                'images': uploaded_files
            }
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded for {person_name}")
    
    if st.button("🚀 Train Model", type="primary"):
        if len(training_data) == 0:
            st.error("Please upload at least one image for training!")
        else:
            with st.spinner("Training model... This may take a moment."):
                all_images = []
                all_labels = []
                label_mapping = {}
                
                # Prepare data
                for label, data in training_data.items():
                    label_mapping[label] = data['name']
                    
                    for uploaded_file in data['images']:
                        # Read image
                        image = Image.open(uploaded_file)
                        image_np = np.array(image)
                        
                        # Convert RGB to BGR for OpenCV
                        if len(image_np.shape) == 3:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        all_images.append(image_np)
                        all_labels.append(label)
                
                # Extract faces and train
                faces, face_labels = prepare_training_data(all_images, all_labels)
                
                if len(faces) == 0:
                    st.error("No faces detected in the uploaded images! Please upload images with clear faces.")
                else:
                    recognizer = train_lbph_model(faces, face_labels)
                    
                    # Save to session state
                    st.session_state.trained_model = recognizer
                    st.session_state.label_mapping = label_mapping
                    
                    st.success(f"✅ Model trained successfully with {len(faces)} face samples!")
                    st.balloons()
                    
                    # Display training summary
                    st.subheader("Training Summary")
                    for label, name in label_mapping.items():
                        count = face_labels.count(label)
                        st.write(f"- **{name}**: {count} face samples")

# Recognize Faces Mode
elif mode == "Recognize Faces":
    st.header("🔍 Recognize Faces")
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train the model first in 'Train Model' mode!")
    else:
        st.success("✅ Model is ready for recognition!")
        
        # Display trained people
        st.subheader("Trained People:")
        for label, name in st.session_state.label_mapping.items():
            st.write(f"- {name}")
        
        st.markdown("---")
        
        confidence_threshold = st.slider(
            "Confidence Threshold (lower = stricter):", 
            min_value=0, 
            max_value=100, 
            value=70,
            help="Lower values require more confident matches"
        )
        
        uploaded_file = st.file_uploader(
            "Upload an image to recognize:", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            # Read and display original image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Perform recognition
            with st.spinner("Recognizing faces..."):
                results = recognize_face(
                    image_np, 
                    st.session_state.trained_model, 
                    cascade, 
                    st.session_state.label_mapping,
                    confidence_threshold
                )
                
                # Draw results
                output_image = draw_results(image_np, results)
                output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Recognition Results")
                st.image(output_image_rgb, use_container_width=True)
            
            # Display results summary
            st.markdown("---")
            st.subheader("Detection Summary")
            
            if len(results) == 0:
                st.info("No faces detected in the image.")
            else:
                for idx, result in enumerate(results, 1):
                    status = "✅" if result['name'] != 'Unknown' else "❌"
                    st.write(f"{status} **Face {idx}**: {result['name']} (Confidence: {result['confidence']:.2f})")

# Footer
st.markdown("---")
st.markdown("""
### About This App
This face recognition application uses:
- **Haar Cascades**: For face detection
- **LBPH (Local Binary Patterns Histograms)**: For face recognition

**Tips for best results:**
- Use clear, front-facing photos for training
- Upload multiple images per person (3-5 recommended)
- Ensure good lighting in training images
- Use similar lighting conditions for recognition
""")
