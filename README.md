# Hu Moment Detection

This project implements letter detection based on Hu moments using OpenCV. It compares pre-calculated Hu moments with test images or video frames to identify the letter present in the image.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/hu-moment-detection.git
   cd hu-moment-detection
   ```

2. **Install the required dependencies:**
   Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare pre-calculated Hu moments:**
   Ensure you have pre-calculated Hu moments stored in JSON files for comparison:
   - `hu_moments_h.json`
   - `hu_moments_s.json`
   - `hu_moments_u.json`

   These files should contain the Hu moments for the letters "H", "S", and "U", respectively.

2. **Run the detection script:**
   To start letter detection from a video stream (e.g., from a mobile phone IP camera), run the script:
   ```bash
   python hu_moments.py
   ```

   The script will process frames and output the detected letter in real-time.

3. **Using with video capture:**
   The script is designed to capture video from your phone’s camera. Replace the IP in the script with your phone’s IP address:
   ```python
   phone_ip = "192.168.2.219"  # Replace with your phone's IP address
   ```

4. **Output:**
   The detected letter will be displayed on the video stream, with a similarity score for each letter (H, S, U). The letter with the highest similarity score is shown.

## Project Structure

- `hu_moments.py`: The main script that processes video frames and performs letter detection.
- `hu_moments_h.json`, `hu_moments_s.json`, `hu_moments_u.json`: JSON files containing pre-calculated Hu moments for letters H, S, and U.
- `requirements.txt`: A file that lists the required Python packages.

## How It Works

1. **Hu Moment Calculation:**
   - The Hu moments of the test image are calculated from its contours.
   
2. **Similarity Metric:**
   - The calculated Hu moments are compared with pre-calculated values to determine the similarity.

3. **Letter Estimation:**
   - Based on the similarity scores, the script estimates the letter (H, S, or U) most likely to match.

## Example Usage

```python
from hu_moments import PrecalculatedHuLoader, HuMomentCalculator, VideoProcessor

# Load pre-calculated Hu moments
hu_loader = PrecalculatedHuLoader()
hu_moments_h = hu_loader.load_hu_moments_from_json('hu_moments_h.json')
hu_moments_s = hu_loader.load_hu_moments_from_json('hu_moments_s.json')
hu_moments_u = hu_loader.load_hu_moments_from_json('hu_moments_u.json')

# Initialize HuMomentCalculator and VideoProcessor
hu_calculator = HuMomentCalculator()
video_processor = VideoProcessor("http://192.168.2.219:8080/video")

# Start processing video
video_processor.process_video(hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u)
```

## Contributing

Contributions are welcome! Please open issues or pull requests for suggestions or improvements.