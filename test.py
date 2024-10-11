import cv2
import numpy as np
import json
import time
from PIL import Image, ImageTk  # Import Pillow for image handling

# Load pre-calculated Hu moments from JSON files
def load_hu_moments_from_json(filename):
    """Load Hu moments from a JSON file."""
    with open(filename, 'r') as json_file:
        return json.load(json_file)

# Load base Hu moments
hu_moments_h = load_hu_moments_from_json('hu_moments_h.json')
hu_moments_s = load_hu_moments_from_json('hu_moments_s.json')
hu_moments_u = load_hu_moments_from_json('hu_moments_u.json')

def calculate_hu_moments(image):
    """Calculate and return the Hu moments for a given image."""
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    letter_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(letter_contour)
    hu_moments = cv2.HuMoments(moments)
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments_log.flatten()

def calculate_similarity_metric(base_hu, test_hu):
    """Calculate the similarity between two sets of Hu moments."""
    similarities = [
        max((1 - abs(base - test) / abs(base)) * 100, 0)
        for base, test in zip(base_hu, test_hu)
    ]
    return np.mean(similarities)

def compare_with_precalculated_hu(test_hu, hu_moments_data):
    """Compare the test image's Hu moments with pre-calculated Hu moments."""
    results = {}
    for base_image_path, base_hu in hu_moments_data.items():
        similarity = calculate_similarity_metric(np.array(base_hu), test_hu)
        results[base_image_path] = similarity
    return results

def estimate_letter(results_h, results_s, results_u):
    """Estimate the detected letter based on the maximum similarity results."""
    max_h = max(results_h.values(), default=0)
    max_s = max(results_s.values(), default=0)
    max_u = max(results_u.values(), default=0)

    if max_h > max(max_s, max_u):
        return "H"
    elif max_s > max(max_h, max_u):
        return "S"
    elif max_u > max(max_h, max_s):
        return "U"
    else:
        return "None"

def save_results_to_json(results_h, results_s, results_u, filename="output_results.json"):
    """Save the results to a JSON file for later analysis."""
    data = {
        "H": results_h,
        "S": results_s,
        "U": results_u
    }
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def process_frame(frame):
    """Process a single video frame for letter detection."""
    # Increase exposure by multiplying pixel values
    exposure_factor = 2.5  # Adjust this value to increase or decrease exposure
    frame_exposed = cv2.convertScaleAbs(frame, alpha=exposure_factor, beta=0)

    # Convert frame to grayscale for processing
    frame_gray = cv2.cvtColor(frame_exposed, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to detect the letter (inverted)
    _, binary_frame = cv2.threshold(frame_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours (the outlines of the letter)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the bounding box of the largest contour (assuming the letter is the largest object)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Increase the bounding box size by 10 pixels in width and height
        # and adjust the x and y positions accordingly
        x = max(x - 25, 0)  # Ensure x does not go below 0
        y = max(y - 25, 0)  # Ensure y does not go below 0
        w += 50
        h += 50

        # Crop the region around the letter
        letter_frame = binary_frame[y:y+h, x:x+w]

        # Convert the cropped region to a PIL image for displaying
        letter_pil = Image.fromarray(letter_frame)

        # Convert to binary (black and white) and invert the colors
        binary_inverted_image = letter_pil.point(lambda p: 0 if p == 0 else 255, '1')

        # Calculate Hu moments
        test_hu = calculate_hu_moments(letter_frame)

        # Start timer for Hu calculations
        start_time = time.time()

        if test_hu is None:
            print("No valid contour detected.")
            return

        results_h = compare_with_precalculated_hu(test_hu, hu_moments_h)
        results_s = compare_with_precalculated_hu(test_hu, hu_moments_s)
        results_u = compare_with_precalculated_hu(test_hu, hu_moments_u)

        detected_letter = estimate_letter(results_h, results_s, results_u)
        print(f"Detected Letter: {detected_letter}")

        # Calculate and print FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        # Overlay text on the frame
        overlay_text = f"Letter: {detected_letter} | FPS: {fps:.2f}"
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the binary inverted image if needed for debugging
        # cv2.imshow("Binary Inverted", binary_inverted_image)

# Main function to capture video and process frames
def main():
    phone_ip = "192.168.2.219"  # Replace <phone_ip> with your phone's IP address
    video_url = f"http://{phone_ip}:8080/video"
    
    # Start video capture from the phone's IP
    cap = cv2.VideoCapture(video_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        process_frame(frame)

        # Display the frame (optional)
        cv2.imshow('Video Frame', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
