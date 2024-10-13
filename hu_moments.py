import cv2
import numpy as np
import json
import time
from PIL import Image

class PrecalculatedHuLoader:
    @staticmethod
    def load_hu_moments_from_json(filename):
        """
        Load Hu moments from a JSON file.

        Args:
            filename (str): The path to the JSON file containing pre-calculated Hu moments.

        Returns:
            dict: A dictionary containing the Hu moments.
        """
        with open(filename, 'r') as json_file:
            return json.load(json_file)

class HuMomentCalculator:
    @staticmethod
    def calculate_hu_moments(image):
        """
        Calculate and return the Hu moments for a given image.

        Args:
            image (numpy.ndarray): The input image for which to calculate Hu moments.

        Returns:
            numpy.ndarray: The calculated Hu moments, or None if no contours are found.
        """
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        letter_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(letter_contour)
        hu_moments = cv2.HuMoments(moments)
        return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    @staticmethod
    def calculate_similarity(base_hu, test_hu):
        """
        Calculate the similarity between two sets of Hu moments.

        Args:
            base_hu (numpy.ndarray): The base Hu moments to compare against.
            test_hu (numpy.ndarray): The Hu moments of the test image.

        Returns:
            float: The similarity percentage between the base and test Hu moments.
        """
        return np.mean([
            max((1 - abs(base - test) / abs(base)) * 100, 0)
            for base, test in zip(base_hu, test_hu)
        ])

class LetterEstimator:
    @staticmethod
    def estimate_letter(results_h, results_s, results_u):
        """
        Estimate the detected letter based on the maximum similarity results.

        Args:
            results_h (dict): The similarity results for letter 'H'.
            results_s (dict): The similarity results for letter 'S'.
            results_u (dict): The similarity results for letter 'U'.

        Returns:
            str: The detected letter ('H', 'S', 'U', or 'None' if no valid letter is detected).
        """
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
        """
        Process a single video frame for letter detection.

        Args:
            frame (numpy.ndarray): The input video frame.
            hu_calculator (HuMomentCalculator): The calculator instance for computing Hu moments.
            hu_moments_h (numpy.ndarray): Pre-calculated Hu moments for letter 'H'.
            hu_moments_s (numpy.ndarray): Pre-calculated Hu moments for letter 'S'.
            hu_moments_u (numpy.ndarray): Pre-calculated Hu moments for letter 'U'.

        Returns:
            numpy.ndarray: The Hu moments of the detected letter in the frame, or None if not detected.
        """
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

        return test_hu

class VideoProcessor:
    def __init__(self, video_url):
        """
        Initialize the VideoProcessor with the video URL.

        Args:
            video_url (str): The URL or file path of the video to process.
        """
        self.video_url = video_url
        self.cap = cv2.VideoCapture(self.video_url)

    def process_video(self, hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u):
        """
        Capture and process each frame from the video for letter detection.

        Args:
            hu_calculator (HuMomentCalculator): The calculator instance for computing Hu moments.
            hu_moments_h (numpy.ndarray): Pre-calculated Hu moments for letter 'H'.
            hu_moments_s (numpy.ndarray): Pre-calculated Hu moments for letter 'S'.
            hu_moments_u (numpy.ndarray): Pre-calculated Hu moments for letter 'U'.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            test_hu = FrameProcessor.process_frame(frame, hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u)
            if test_hu is None:
                continue

            results_h = hu_calculator.calculate_similarity(hu_moments_h, test_hu)
            results_s = hu_calculator.calculate_similarity(hu_moments_s, test_hu)
            results_u = hu_calculator.calculate_similarity(hu_moments_u, test_hu)

            detected_letter = LetterEstimator.estimate_letter(results_h, results_s, results_u)
            print(f"Detected Letter: {detected_letter}")

            cv2.imshow('Video Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    hu_loader = PrecalculatedHuLoader()
    hu_moments_h = hu_loader.load_hu_moments_from_json('hu_moments_h.json')
    hu_moments_s = hu_loader.load_hu_moments_from_json('hu_moments_s.json')
    hu_moments_u = hu_loader.load_hu_moments_from_json('hu_moments_u.json')

    hu_calculator = HuMomentCalculator()
    video_processor = VideoProcessor("http://192.168.2.219:8080/video")
    video_processor.process_video(hu_calculator, hu_moments_h, hu_moments_s, hu_moments_u)
