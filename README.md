# Hu Moment Detection from Video Streams

This project processes video frames to detect letters based on their Hu moments. Hu moments are scale, rotation, and translation invariant properties used to uniquely identify shapes. In this project, Hu moments are calculated for letters "H", "S", and "U" for real-time detection in video streams.

## Disclaimer

This is not a professional project, but rather a personal research effort to explore the use of Hu moments in shape detection. Contributions are more than welcome! If you would like to improve the code, add new features, or suggest enhancements, feel free to fork the repository and submit a pull request.

## Features

- **Input Video from Drive**: Load videos directly from your computer for processing.
- **IP Webcam Support**: Stream video from your IP webcam by installing an app such as "IP Webcam" from the Android Play Store and use the URL (e.g., `http://<ip>:8080/video`) to feed the video stream into the program.
- **Extended Control Feature**: This mode allows you to move back and forth between frames using keyboard controls, pause the video, and manually control the playback.
- **Pre-calculated Hu Moments**: The program compares the real-time frame data with pre-calculated Hu moments for efficient letter detection.

## Requirements

Make sure you have the following Python packages installed:

```bash
pip install -r requirements.txt
```

## Getting Started

### Pre-calculating Hu Moments

To begin, you need to have pre-calculated Hu moments for the letters you want to detect. This project assumes you are comparing frames against three sets of letters: "H", "S", and "U". 

The pre-calculated Hu moments should be stored in JSON files as follows:

```json
{
    "path/to/image1.png": [hu_moment_value_1, hu_moment_value_2, ..., hu_moment_value_7],
    "path/to/image2.png": [hu_moment_value_1, hu_moment_value_2, ..., hu_moment_value_7]
}
```

Ensure you have the following JSON files:

- `hu_moments_h.json`: Hu moments for letter "H".
- `hu_moments_s.json`: Hu moments for letter "S".
- `hu_moments_u.json`: Hu moments for letter "U".

These files will be loaded into the program to compare Hu moments against the video stream.

### Usage

#### Running with Local Video

To run the program with a video file from your computer, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FinalZenix/opencv-hu-moments.git
   cd opencv-hu-moments
   ```

2. **Prepare your Hu moments JSON files** and place them in the root directory.

3. **Run the script** with your local video file:

   ```python
   python hu_moments.py
   ```

   By default, the program is set to load the video from the `./test_video.mp4` file. If you want to use your own video file, edit the line in `hu_moments.py` to point to your video:

   ```python
   video_processor = VideoProcessor("./path_to_your_video.mp4")
   ```

#### Running with an IP Webcam

If you prefer to use a mobile device as your webcam, follow these steps:

1. **Install the "IP Webcam" app** from the [Android Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam).

2. **Start the IP Webcam app** and note the URL provided by the app (e.g., `http://<ip>:8080/video`).

3. In the code, replace the video path with the URL from the IP Webcam:

   ```python
   video_processor = VideoProcessor("http://<ip>:8080/video")
   ```

   This will stream video from your IP webcam into the program.

### Keyboard Controls

If you enable **extended controls**, you can manually control the video playback using the following keys:

- **k**: Pause and resume the video.
- **Left Arrow Key**: Go back one frame.
- **Right Arrow Key**: Go forward one frame.
- **Q**: Quit the video processing and close all windows.

### Example Output

When processing the video, the program will display the following information:

1. **Detected Letters**: The letter detected in the current frame (either "H", "S", or "U") will be printed in the terminal.
2. **Frame Info**: The current frame number and total frames will be displayed on the video.

## Project Structure

```
opencv-hu-moments/
│
├── hu_moments_h.json           # JSON file with Hu moments for letter "H"
├── hu_moments_s.json           # JSON file with Hu moments for letter "S"
├── hu_moments_u.json           # JSON file with Hu moments for letter "U"
├── hu_moments.py                     # Main script to run the video processing
├── README.md                   # This file
└── requirements.txt            # List of dependencies
```

## How It Works

1. **Loading Hu Moments**: The program loads pre-calculated Hu moments from JSON files using the `PrecalculatedHuLoader` class.
2. **Frame Processing**: Each video frame is processed by:
   - Converting the frame to grayscale.
   - Applying a binary threshold and extracting the largest contour (letter).
   - Calculating the Hu moments for the contour and comparing them to the pre-calculated moments.
3. **Similarity Metric**: The similarity between the frame's Hu moments and the pre-calculated moments is calculated as a percentage. The highest similarity score determines the detected letter.
4. **Extended Controls**: If enabled, the user can pause the video, skip frames, or manually control the playback speed.

## Example Usage

To process a video with extended controls enabled, you can modify the `hu_moments.py` as follows:

```python
if __name__ == "__main__":
    hu_loader = PrecalculatedHuLoader()
    hu_moments_h = hu_loader.load_hu_moments_from_json('hu_moments_h.json')
    hu_moments_s = hu_loader.load_hu_moments_from_json('hu_moments_s.json')
    hu_moments_u = hu_loader.load_hu_moments_from_json('hu_moments_u.json')

    hu_calculator = HuMomentCalculator()
    
    # Using a local video file
    video_processor = VideoProcessor("./path_to_your_video.mp4", use_extended_controls=False)
    
    # Or using an IP Webcam stream
    # video_processor = VideoProcessor("http://<ip>:8080/video", use_extended_controls=False)
    
    video_processor.process_video(hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u)
```

In this example, you can either provide a local video file or use a stream from an IP webcam. Set `use_extended_controls` to `True` for manual control of the video playback.

## License

This project is licensed under the MIT License.
