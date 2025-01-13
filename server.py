import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import time
from collections import deque
import plotly.graph_objects as go
from PIL import Image
import logging
import datetime

class EmotionDetectionSystem:
    def __init__(self, model_path, buffer_size=30):
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ["Boredom", "Confusion", "Engagement", "Frustration"]
        self.image_height = 196
        self.image_width = 196
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.emotion_history = {emotion: deque(maxlen=buffer_size) for emotion in self.emotion_labels}
        self.colors = {
            "Boredom": "#ef4444",      # Red
            "Confusion": "#3b82f6",     # Blue
            "Engagement": "#22c55e",    # Green
            "Frustration": "#a855f7"    # Purple
        }
        
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
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
                
                color = tuple(int(self.colors[stable_emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), color, 2)
                label = f"{stable_emotion}: {confidence*100:.1f}%"
                cv2.putText(rgb_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                results.append({
                    "emotion": stable_emotion,
                    "confidence": confidence,
                    "predictions": predictions[0].tolist()
                })
        
        return rgb_frame, results

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
    
    # Add custom CSS
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
    
    # Initialize session state variables
    if 'detector' not in st.session_state:
        st.session_state.detector = EmotionDetectionSystem("Transfer-Learning-v1.keras")
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    st.markdown('<p class="emoji-header">🎭 Real-time Emotion Detection System</p>', unsafe_allow_html=True)
    
    # Create fixed containers for the layout
    button_col1, button_col2 = st.columns([1, 5])
    with button_col1:
        if not st.session_state.camera_running:
            if st.button('Start Camera'):
                st.session_state.cap = cv2.VideoCapture(0)
                if st.session_state.cap.isOpened():
                    st.session_state.camera_running = True
                else:
                    st.error("Failed to open webcam")
        else:
            if st.button('Stop Camera'):
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                st.session_state.camera_running = False
                st.session_state.cap = None
                st.experimental_rerun()

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fixed container for video
        video_container = st.empty()
    
    with col2:
        # Fixed containers for metrics and chart
        metrics_container = st.empty()
        chart_container = st.empty()

    if st.session_state.camera_running and st.session_state.cap is not None:
        try:
            while st.session_state.camera_running:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                processed_frame, results = st.session_state.detector.process_frame(frame)
                
                # Update video feed
                video_container.image(processed_frame, channels="RGB", use_column_width=True)
                
                # Update metrics
                if results:
                    latest_result = results[-1]
                    metric_cols = metrics_container.columns(len(st.session_state.detector.emotion_labels))
                    for col, emotion in zip(metric_cols, st.session_state.detector.emotion_labels):
                        confidence = latest_result["predictions"][
                            st.session_state.detector.emotion_labels.index(emotion)
                        ]
                        col.metric(
                            emotion,
                            f"{confidence*100:.1f}%",
                            delta=None,
                            delta_color="normal"
                        )
                
                # Update chart
                chart = create_emotion_chart(st.session_state.detector.emotion_history)
                chart_container.plotly_chart(chart, use_container_width=True)
                
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.camera_running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
            st.session_state.cap = None
    else:
        # Display placeholder content when camera is not running
        video_container.markdown("Camera is not running. Click 'Start Camera' to begin.")
        metrics_container.markdown("Waiting for emotion detection...")
        chart_container.markdown("Emotion trends will appear here...")

if __name__ == "__main__":
    main()