import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque
import plotly.graph_objects as go
from PIL import Image
import logging
import datetime
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetectionSystem:
    def __init__(self, model_path, buffer_size=30):
        try:
            self.model = load_model(model_path)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.emotion_labels = ["Boredom", "Confusion", "Engagement", "Frustration"]
            self.image_height = 196
            self.image_width = 196
            self.prediction_buffer = deque(maxlen=buffer_size)
            self.emotion_history = {emotion: deque(maxlen=buffer_size) for emotion in self.emotion_labels}
            self.colors = {
                "Boredom": "#ef4444",
                "Confusion": "#3b82f6",
                "Engagement": "#22c55e",
                "Frustration": "#a855f7"
            }
            logger.info("EmotionDetectionSystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing EmotionDetectionSystem: {str(e)}")
            raise

    def process_frame(self, frame_data):
        try:
            # Convert the frame data to a format OpenCV can process
            image = Image.open(frame_data)
            frame = np.array(image)
            
            # Convert to RGB if necessary
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            results = []
            frame_rgb = frame.copy()
            
            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (self.image_width, self.image_height))
                normalized_face = resized_face / 255.0
                input_face = normalized_face.reshape(1, self.image_height, self.image_width, 1)
                
                predictions = self.model.predict(input_face, verbose=0)
                emotion_index = np.argmax(predictions)
                confidence = np.max(predictions)
                
                self.prediction_buffer.append(emotion_index)
                
                for emotion, pred in zip(self.emotion_labels, predictions[0]):
                    self.emotion_history[emotion].append(pred)
                
                if len(self.prediction_buffer) >= 5:
                    emotion_counts = np.bincount(list(self.prediction_buffer)[-5:])
                    stable_emotion_index = np.argmax(emotion_counts)
                    stable_emotion = self.emotion_labels[stable_emotion_index]
                    
                    # Convert color from hex to RGB
                    color = tuple(int(self.colors[stable_emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
                    label = f"{stable_emotion}: {confidence*100:.1f}%"
                    cv2.putText(frame_rgb, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    results.append({
                        "emotion": stable_emotion,
                        "confidence": confidence,
                        "predictions": predictions[0].tolist()
                    })
            
            return frame_rgb, results
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            raise

def create_emotion_chart(emotion_history):
    fig = go.Figure()
    
    for emotion, values in emotion_history.items():
        if len(values) > 0:
            fig.add_trace(go.Scatter(
                y=list(values),
                name=emotion,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Emotion Trends",
        xaxis_title="Time",
        yaxis_title="Confidence",
        yaxis_range=[0, 1],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def main():
    st.set_page_config(page_title="Emotion Detection System", layout="wide")
    
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
        }
        .emoji-header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = EmotionDetectionSystem("Transfer-Learning-v1.keras")
        except Exception as e:
            st.error(f"Failed to initialize detector: {str(e)}")
            return

    st.markdown('<p class="emoji-header">🎭 Real-time Emotion Detection System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_input = st.camera_input("Take a picture")
        if camera_input is not None:
            try:
                processed_frame, results = st.session_state.detector.process_frame(camera_input)
                st.image(processed_frame, channels="RGB", use_column_width=True)
                
                if results:
                    latest_result = results[-1]
                    cols = st.columns(len(st.session_state.detector.emotion_labels))
                    for col, emotion in zip(cols, st.session_state.detector.emotion_labels):
                        confidence = latest_result["predictions"][
                            st.session_state.detector.emotion_labels.index(emotion)
                        ]
                        col.metric(
                            emotion,
                            f"{confidence*100:.1f}%",
                            delta=None,
                            delta_color="normal"
                        )
                
                chart = create_emotion_chart(st.session_state.detector.emotion_history)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.markdown("""
        ### How to use:
        1. Click the "Take a picture" button
        2. Allow camera access if prompted
        3. Take a photo
        4. View the emotion detection results
        
        The system will detect:
        - Boredom
        - Confusion
        - Engagement
        - Frustration
        
        The graph shows emotion confidence trends over time.
        """)

if __name__ == "__main__":
    main()
