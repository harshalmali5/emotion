import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

class EmotionDetectionSystem:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ["Boredom", "Confusion", "Engagement", "Frustration"]
        self.image_height = 196
        self.image_width = 196

    def process_frame(self, frame_data):
        image = Image.open(frame_data)
        frame = np.array(image)
        
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (self.image_width, self.image_height))
            normalized_face = resized_face / 255.0
            input_face = normalized_face.reshape(1, self.image_height, self.image_width, 1)
            
            predictions = self.model.predict(input_face, verbose=0)
            emotion_index = np.argmax(predictions)
            confidence = np.max(predictions)
            emotion = self.emotion_labels[emotion_index]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

def main():
    st.title('Emotion Detection System')
    
    if 'detector' not in st.session_state:
        st.session_state.detector = EmotionDetectionSystem("Transfer-Learning-v1.keras")

    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None:
        try:
            processed_frame = st.session_state.detector.process_frame(camera_input)
            st.image(processed_frame, channels="RGB")
            st.success("Image processed successfully!")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
