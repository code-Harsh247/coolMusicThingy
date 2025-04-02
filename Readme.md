# Hand Gesture Chord Player

A computer vision application that allows you to play guitar chords by showing different finger gestures to your webcam.

## Overview

This application uses your webcam to detect hand gestures and plays corresponding guitar chord sounds. By forming different finger combinations, you can trigger various chords without touching your computer or instrument.

## Features

- Real-time hand tracking and finger position detection
- Intuitive gesture-to-chord mapping
- Visual feedback showing detected finger states
- Optimized performance with frame skipping
- Multi-threaded audio playback

## Chord Gestures

| Gesture | Chord |
|---------|-------|
| Index finger only | C |
| Index + Middle | D |
| Index + Middle + Ring | G |
| All fingers except thumb | A |
| Thumb + Index | E |
| Thumb + Middle | F |
| Thumb + Ring | B |
| Thumb + Pinky | Em |
| Middle + Ring + Pinky | Am |
| All fingers | Bm7 |

## Requirements

- Python 3.7+
- Webcam
- The following Python packages (see requirements.txt):
  - OpenCV
  - MediaPipe
  - PyGame
  - NumPy

## Installation

1. Clone this repository or download the source code
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have a folder named "sounds" with the following WAV files:
   - C.wav, D.wav, E.wav, F.wav, G.wav, A.wav, B.wav, Em.wav, Am.wav, Bm7.wav

## Usage

1. Run the program:
   ```
   python hand_chord_player.py
   ```

2. Show your hand to the camera with different finger combinations to play chords
3. Press 'q' to exit the program

## How It Works

1. The application captures video from your webcam
2. MediaPipe's hand detection model identifies hand landmarks
3. Finger positions are analyzed to determine which fingers are up/down
4. When a recognized gesture is detected, the corresponding chord sound is played
5. Visual feedback shows the current finger state and active chord

## Performance Optimization

- Frame skipping for smoother performance on slower systems
- Multi-threaded audio playback to prevent video lag
- Optimized hand landmark detection parameters
- Gesture stability checking to prevent accidental triggers

## Customization

You can modify the gesture-to-chord mapping in the `gesture_map` dictionary to create your own custom chord combinations.

## Limitations

- Works best in good lighting conditions
- May have difficulty with very fast movements
- Requires clear view of all fingers

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe for the hand tracking technology
- PyGame for audio playback capabilities