import cv2
import mediapipe as mp
import time
import pygame
import numpy as np
from collections import deque
import threading

# Initialize pygame mixer
pygame.mixer.init()

# Expanded chord dictionary
chord_sounds = {
    "C": pygame.mixer.Sound("sounds/C.wav"),
    "D": pygame.mixer.Sound("sounds/D.wav"),
    "E": pygame.mixer.Sound("sounds/E.wav"),
    "F": pygame.mixer.Sound("sounds/F.wav"),
    "G": pygame.mixer.Sound("sounds/G.wav"),
    "A": pygame.mixer.Sound("sounds/A.wav"),
    "B": pygame.mixer.Sound("sounds/B.wav"),
    "Em": pygame.mixer.Sound("sounds/Em.wav"),
    "Am": pygame.mixer.Sound("sounds/Am.wav"),
    "Bm7": pygame.mixer.Sound("sounds/Bm7.wav"),
}

# Audio playback thread
def play_chord_thread(chord):
    """Play a chord at maximum volume in a separate thread"""
    if chord in chord_sounds:
        chord_sounds[chord].set_volume(1.0)  # Set to maximum volume
        chord_sounds[chord].play()

# Gesture history for stability
class GestureHistory:
    def __init__(self, size=5):
        self.history = deque(maxlen=size)
        self.current_gesture = None
        self.confirmed = False
        self.confirm_count = 0
        self.confirm_threshold = 2  # Reduced for faster response
    
    def add(self, gesture):
        self.history.append(gesture)
        
        # Check if we have a stable gesture
        if len(self.history) == self.history.maxlen:
            most_common = max(set(self.history), key=self.history.count)
            
            # If most common gesture appears in at least 50% of history (reduced threshold)
            if self.history.count(most_common) >= self.history.maxlen * 0.5:
                # If this is a new gesture
                if most_common != self.current_gesture:
                    self.current_gesture = most_common
                    self.confirmed = False
                    self.confirm_count = 1
                else:
                    # Same gesture, increment confirmation counter
                    self.confirm_count += 1
                    if self.confirm_count >= self.confirm_threshold:
                        self.confirmed = True
                return most_common, self.confirmed
        
        return None, False

# Mapping from finger configurations to chords
gesture_map = {
    # Format: [thumb, index, middle, ring, pinky] -> chord name
    (0, 1, 0, 0, 0): "C",    # Index only
    (0, 1, 1, 0, 0): "D",    # Index + Middle
    (0, 1, 1, 1, 0): "G",    # Index + Middle + Ring
    (0, 1, 1, 1, 1): "A",    # All fingers except thumb
    (1, 1, 0, 0, 0): "E",    # Thumb + Index
    (1, 0, 1, 0, 0): "F",    # Thumb + Middle
    (1, 0, 0, 1, 0): "B",    # Thumb + Ring
    (1, 0, 0, 0, 1): "Em",   # Thumb + Pinky
    (0, 0, 1, 1, 1): "Am",   # Middle + Ring + Pinky
    (1, 1, 1, 1, 1): "Bm7",  # All fingers
}

def get_fingers_state(hand_landmarks, img_shape):
    """Get the state of each finger (extended or not)"""
    h, w, _ = img_shape
    lm_list = []
    
    # Extract landmark positions (optimized to only get the ones we need)
    for id in [0, 3, 4, 5, 6, 8, 10, 12, 14, 16, 17, 18, 20]:
        lm = hand_landmarks.landmark[id]
        cx, cy = int(lm.x * w), int(lm.y * h)
        lm_list.append([id, cx, cy])
    
    # Create a mapping for faster lookup
    lm_dict = {item[0]: item for item in lm_list}
    
    fingers = []
    
    # Thumb: Depending on hand direction, compare x position (FIXED - inverted logic)
    if lm_dict[5][1] > lm_dict[17][1]:  # Left hand
        fingers.append(1 if lm_dict[4][1] > lm_dict[3][1] else 0)
    else:  # Right hand
        fingers.append(1 if lm_dict[4][1] < lm_dict[3][1] else 0)
    
    # 4 Fingers: Check if tip is higher than the middle joint
    finger_tips = [8, 12, 16, 20]
    finger_mids = [6, 10, 14, 18]
    
    for tip_id, mid_id in zip(finger_tips, finger_mids):
        if tip_id in lm_dict and mid_id in lm_dict:
            fingers.append(1 if lm_dict[tip_id][2] < lm_dict[mid_id][2] else 0)
        else:
            fingers.append(0)  # Default to closed if landmark missing
    
    return fingers

def draw_binary_finger_state(img, fingers):
    """Draw technical binary representation of finger states"""
    h, w, _ = img.shape
    
    if not fingers:
        return
    
    # Position for binary display
    start_x = 10
    start_y = 40
    
    # Draw binary header text
    cv2.putText(img, "FINGER STATE [T,I,M,R,P]", (start_x, start_y - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Draw binary digits directly without boxes
    binary_string = ''.join([str(f) for f in fingers])
    for i, digit in enumerate(binary_string):
        # Position for this digit
        pos_x = start_x + i * 30 + 10
        
        # Color based on value (green/red)
        color = (0, 220, 0) if digit == '1' else (0, 0, 220)
        
        # Draw the binary value
        cv2.putText(img, digit, (pos_x, start_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_minimal_ui(img, chord=None, fps=0, fingers=None):
    """Draw a minimal, clean UI without any boxes or extras"""
    h, w, _ = img.shape
    
    # Display FPS counter in corner
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(img, fps_text, (w - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    # Display active chord
    if chord:
        cv2.putText(img, chord, (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)
    
    # Display finger states without boxes
    if fingers:
        start_y = 140
        finger_names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        
        for i, (finger, name) in enumerate(zip(fingers, finger_names)):
            # Position
            pos_y = start_y + i * 25
            
            # Status color
            status_color = (0, 220, 0) if finger == 1 else (0, 0, 220)
            
            # Finger name and status
            status = "ACTIVE" if finger == 1 else "INACTIVE"
            text = f"{name}: {status}"
            cv2.putText(img, text, (10, pos_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    # Draw binary representation of finger states
    if fingers:
        draw_binary_finger_state(img, fingers)
    
    return img

def main():
    # Camera setup
    cap = cv2.VideoCapture(0)
    
    # Set reasonable resolution for balance of performance and visibility
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Track the last time we played a chord
    last_play_time = 0
    
    # MediaPipe hands setup with optimized parameters
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Create clean, technical drawing specs for hand landmarks
    landmark_drawing_spec = mp_draw.DrawingSpec(
        color=(0, 180, 255),  # Modern blue dots
        thickness=1,
        circle_radius=3
    )
    connection_drawing_spec = mp_draw.DrawingSpec(
        color=(255, 255, 255),  # White connections
        thickness=1,
        circle_radius=1
    )
    
    # Performance tracking
    p_time = 0
    frame_count = 0
    skipped_frames = 0
    process_every_n_frames = 2
    
    # Gesture tracking
    gesture_history = GestureHistory(size=5)
    last_chord = ""
    last_play_time = 0
    
    # Performance monitoring
    fps_history = deque(maxlen=30)
    
    while True:
        frame_count += 1
        
        # Skip frames for better performance
        if frame_count % process_every_n_frames != 0:
            skipped_frames += 1
            # Still need to read the frame to keep the camera buffer clear
            cap.read()
            # But we can show the last processed frame if we have it
            if 'display_img' in locals():
                cv2.imshow("Cool Music Thingy", display_img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            continue
        
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue
        
        # Mirror the image for more intuitive control
        img = cv2.flip(img, 1)
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = hands.process(img_rgb)
        
        # FPS calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time) if c_time != p_time else 30
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        p_time = c_time
        
        # Initialize finger states
        fingers = None
        current_gesture = None
        
        # Clear chord if no hands detected
        if not results.multi_hand_landmarks:
            if last_chord:
                last_chord = ""
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            # Use the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks with clean styling
            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec, connection_drawing_spec
            )
            
            # Get finger states
            fingers = get_fingers_state(hand_landmarks, img.shape)
            if fingers:
                # Get gesture tuple for mapping
                current_gesture = tuple(fingers)
                
                # Add to history for stability
                chord, confirmed = gesture_history.add(current_gesture)
                
                # If gesture is stable and maps to a chord
                if confirmed and current_gesture in gesture_map:
                    chord_name = gesture_map[current_gesture]
                    current_time = time.time()
                    
                    # Allow replaying the same chord after a cooldown period
                    if chord_name != last_chord or (current_time - last_play_time) > 2:
                        # Start audio playback in a separate thread with max volume
                        threading.Thread(
                            target=play_chord_thread, 
                            args=(chord_name,)
                        ).start()
                        last_chord = chord_name
                        last_play_time = current_time
        
        # Apply the minimal UI
        img = draw_minimal_ui(
            img, 
            chord=last_chord, 
            fps=avg_fps, 
            fingers=fingers
        )
        
        # Save the processed frame for reuse in skipped frames
        display_img = img.copy()
        
        # Show the image
        cv2.imshow("Hand Chord Player", img)
        
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"Performance: Processed {frame_count-skipped_frames} frames, skipped {skipped_frames} frames")

if __name__ == "__main__":
    main()