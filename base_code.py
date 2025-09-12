
import cv2 as cv
import numpy as np
import hashlib
import random
import os

# ============================================================
# Quantum-Inspired Key Generator (Webcam + Shapes)
# ============================================================
# Working:
# This code does NOT use real quantum mechanics hardware.
# Instead, it SIMULATES the "feel" of a quantum measurement:
#   - Random choice of a measurement basis (circle, rect, ellipse)
#   - Capturing unpredictable values (webcam pixels + sensor noise)
#   - "Collapsing" those values into a cryptographic key via hashing
#
# Real quantum random number generators use photons / quantum states.
# Here we substitute with webcam entropy + randomness to simulate a related idea.
# ============================================================


# -------------------------------
# Measurement Basis (our "quantum" shapes)
# -------------------------------
class Basis:
    def __init__(self, kind, center, size):
        self.kind = kind      # the "basis" type: rect, circle, or ellipse
        self.center = center  # (x, y) position of basis
        self.size = size      # size parameter (radius / half-length)


# -------------------------------
# Draw the chosen basis + mask it
# -------------------------------
def draw_basis_and_mask(image, basis):
    """
    Given an image and a chosen "measurement basis" (shape),
    draw the shape outline on the image for visualization,
    and build a MASK of that region (white pixels = inside shape).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y = basis.center
    s = basis.size

    # Rectangular measurement basis
    if basis.kind == "rect":
        top_left = (x - s, y - s)
        bottom_right = (x + s, y + s)
        cv.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
        cv.rectangle(mask, top_left, bottom_right, 255, -1)

    # Circular measurement basis
    elif basis.kind == "circle":
        cv.circle(image, (x, y), s, (0, 255, 255), 2)
        cv.circle(mask, (x, y), s, 255, -1)

    # Elliptical measurement basis
    elif basis.kind == "ellipse":
        cv.ellipse(image, (x, y), (s, s // 2), 0, 0, 360, (255, 0, 0), 2)
        cv.ellipse(mask, (x, y), (s, s // 2), 0, 0, 360, 255, -1)

    return image, mask


# -------------------------------
# Choose a random basis (quantum analogy: random measurement)
# -------------------------------
def random_basis(h, w):
    """
    Randomly choose a measurement basis.
    In real quantum mechanics, we randomly choose measurement
    directions (e.g. photon polarization at 0°, 45°, 90°).
    Here we mimic that idea by choosing between three shape "bases".
    """
    kind = random.choice(["rect", "circle", "ellipse"])
    x = random.randint(100, w - 100)
    y = random.randint(100, h - 100)
    size = random.randint(30, 80)
    return Basis(kind, (x, y), size)


# -------------------------------
# Collapse measurement → Key
# -------------------------------
def measure_wavefunction(gray, mask):
    """
    Quantum analogy:
    - The webcam frame is our 'wavefunction' (uncertain, noisy, full of possibilities).
    - The mask (basis) selects one way of measuring it.
    - The pixels inside the mask are the measurement outcome.
    - We 'collapse' the measurement into a single value by hashing.

    Returns:
        A SHA-256 hex digest (the "quantum key").
    """
    measurement = gray[mask == 255]  # select pixels inside shape (our "collapse")
    if measurement.size == 0:
        return None

    # Salt = additional randomness (to mimic inherent uncertainty)
    salt = os.urandom(4)
    data = measurement.tobytes() + salt

    # Collapse into a deterministic but unpredictable key
    digest = hashlib.sha256(data).hexdigest()
    return digest


# -------------------------------
# Main Experiment Loop
# -------------------------------
def main():
    print("----------------Welcome to the Quantum Inspired Key Generator (1.0)------------------")
    print("This app still under development. So, please look forward to further updates :D")
    print("--------To close the window, press ESC--------")

    # Start webcam ("our quantum source")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally → mirror effect
        frame = cv.flip(frame, 1)

        # Convert to grayscale → reduce to "measurement intensity"
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Randomly choose a measurement basis (shape)
        h, w = gray.shape
        basis = random_basis(h, w)

        # Draw basis + mask pixels inside it
        frame_with_basis, mask = draw_basis_and_mask(frame.copy(), basis)

        # Collapse wavefunction → get quantum key
        key = measure_wavefunction(gray, mask)

        # Overlay experiment info on the video feed
        if key:
            cv.putText(frame_with_basis, f"Quantum Key: {key[:16]}", 
                       (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.6, (0,255,0), 2)
        cv.putText(frame_with_basis, f"Measurement Basis: {basis.kind}", 
                   (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Show the running experiment window
        cv.imshow("Quantum-Inspired Key Generator", frame_with_basis)

        # Refresh every second, quit with ESC
        if cv.waitKey(400) & 0xFF == 27:
            break

    # Close "lab" when done :D
    cap.release()
    cv.destroyAllWindows()


# -------------------------------
# Run Experiment
# -------------------------------
if __name__ == "__main__":
    main()
