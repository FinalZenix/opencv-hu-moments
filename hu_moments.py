import cv2
import numpy as np
import json
import time
from PIL import Image

class PrecalculatedHuLoader:
    @staticmethod
    def load_hu_moments_from_json(filename):
        with open(filename, 'r') as json_file:
            return json.load(json_file)

class HuMomentCalculator:
    @staticmethod
    def calculate_hu_moments(image):
        """
        Calculate the Hu moments for a given binary image.

        Parameters
        ----------
        image : numpy array
            The input binary image.

        Returns
        -------
        hu_moments_log : numpy array
            The calculated Hu moments in log scale as a 1D numpy array.
        """

        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        letter_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(letter_contour)
        hu_moments = cv2.HuMoments(moments)
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments_log.flatten()

    @staticmethod
    def calculate_similarity_metric(base_hu, test_hu):
        similarities = [
            max((1 - abs(base - test) / abs(base)) * 100, 0)
            for base, test in zip(base_hu, test_hu)
        ]
        return np.mean(similarities)

    @staticmethod
    def compare_with_precalculated_hu(test_hu, hu_moments_data):
        results = {}
        for base_image_path, base_hu in hu_moments_data.items():
            similarity = HuMomentCalculator.calculate_similarity_metric(np.array(base_hu), test_hu)
            results[base_image_path] = similarity
        return results

class LetterEstimator:
    @staticmethod
    def estimate_letter(results_h, results_s, results_u):
        max_h = max(results_h.values(), default=0)
        max_s = max(results_s.values(), default=0)
        max_u = max(results_u.values(), default=0)

        if max_h > max(max_s, max_u):
            return "H"
        elif max_s > max(max_h, max_u):
            return "S"
        elif max_u > max(max_h, max_s):
            return "U"
        return "None"

class FrameProcessor:
    @staticmethod
    def process_frame(frame, hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u):
        exposure_factor = 2.5
        frame_exposed = cv2.convertScaleAbs(frame, alpha=exposure_factor, beta=0)
        frame_gray = cv2.cvtColor(frame_exposed, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(frame_gray, 128, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No valid contour detected.")
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x, y, w, h = max(x - 25, 0), max(y - 25, 0), w + 50, h + 50
        letter_frame = binary_frame[y:y + h, x:x + w]

        test_hu = hu_calculator.calculate_hu_moments(letter_frame)
        if test_hu is None:
            print("No valid contour detected.")
            return None

        results_h = hu_calculator.compare_with_precalculated_hu(test_hu, hu_moments_h)
        results_s = hu_calculator.compare_with_precalculated_hu(test_hu, hu_moments_s)
        results_u = hu_calculator.compare_with_precalculated_hu(test_hu, hu_moments_u)

        detected_letter = LetterEstimator.estimate_letter(results_h, results_s, results_u)
        return detected_letter

class VideoProcessor:
    def __init__(self, video_url):
        self.video_url = video_url
        self.cap = cv2.VideoCapture(self.video_url)

    def process_video(self, hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u):
        # Get the frame rate of the video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        cycle_rate = 0  # Initialize cycle rate
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            start_frame_time = time.time()  # Start time for frame processing
            detected_letter = FrameProcessor.process_frame(frame, hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u)
            if detected_letter:
                print(f"Detected Letter: {detected_letter}")

            frame_count += 1
            elapsed_time = time.time() - start_time

            # Calculate cycle rate every second
            if elapsed_time >= 1:  # Update every second
                cycle_rate = frame_count / elapsed_time
                print(f"Cycle Rate: {cycle_rate:.2f} CPS")  # Log cycle rate
                # Reset for next second
                start_time = time.time()
                frame_count = 0

            # Display frame rate and cycle rate
            display_text = f"FPS: {fps:.2f} | Cycle Rate: {cycle_rate:.2f} CPS"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Video Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update processing time for cycle rate calculation
            processing_time = time.time() - start_frame_time
            print(f"Processing Time: {processing_time:.4f} seconds")  # Optional: Print processing time for each frame

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hu_loader = PrecalculatedHuLoader()
    hu_moments_h = hu_loader.load_hu_moments_from_json('hu_moments_h.json')
    hu_moments_s = hu_loader.load_hu_moments_from_json('hu_moments_s.json')
    hu_moments_u = hu_loader.load_hu_moments_from_json('hu_moments_u.json')

    hu_calculator = HuMomentCalculator()
    video_processor = VideoProcessor("./video.mp4")
    video_processor.process_video(hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u)
