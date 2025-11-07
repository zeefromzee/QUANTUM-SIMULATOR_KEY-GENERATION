import matplotlib
matplotlib.use("TkAgg")
import cv2 as cv
import numpy as np
import random
import time   # used for timestamps when saving screenshots
import csv
import os, secrets, hashlib
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import matplotlib.pyplot as plt
# ============================================================
# Quantum-Inspired Key Generator (Webcam + Shapes)
# ============================================================
#  Disclaimer:
# This is NOT a real quantum random number generator.
# Real QRNGs use photons, detectors, and quantum uncertainty.
# Here we simulate the *idea* using:
#   - Webcam pixel noise (as our entropy source)
#   - Random shape "bases" (rect, circle, ellipse)
#   - A SHA-256 hash to collapse measurements into keys
#
# Extra features:
#   - Pause/Resume video with 'P'
#   - Take screenshots with 'S'
#   - Exit program with 'ESC'
# ============================================================


# -------------------------------
# Basis Class (represents measurement basis)
# -------------------------------
class Basis:
    def __init__(self, kind, center, size):
        self.kind = kind      # the type of shape (rect, circle, ellipse)
        self.center = center  # (x, y) center point of the shape
        self.size = size      # size of the shape (radius or half-side length)


# -------------------------------k
# Draw basis on image + build mask
# -------------------------------
def draw_basis_and_mask(image, basis):
    """
    Given an image and a chosen measurement basis:
    - Draw the shape on the frame for visualization
    - Create a MASK (white = inside shape, black = outside)
    """
    h, w = image.shape[:2]                         # frame height, width
    mask = np.zeros((h, w), dtype=np.uint8)        # blank mask (all black)

    x, y = basis.center                            # shape center
    s = basis.size                                 # shape size

    # Case 1: Rectangle basis
    if basis.kind == "rect":
        cv.rectangle(image, (x - s, y - s), (x + s, y + s), (0, 0, 255), 2)   # red outline
        cv.rectangle(mask, (x - s, y - s), (x + s, y + s), 255, -1)           # filled mask

    # Case 2: Circle basis
    elif basis.kind == "circle":
        cv.circle(image, (x, y), s, (0, 255, 255), 2)                         # yellow outline
        cv.circle(mask, (x, y), s, 255, -1)                                   # filled mask

    # Case 3: Ellipse basis
    elif basis.kind == "ellipse":
        cv.ellipse(image, (x, y), (s, s // 2), 0, 0, 360, (255, 0, 0), 2)     # blue outline
        cv.ellipse(mask, (x, y), (s, s // 2), 0, 0, 360, 255, -1)             # filled mask

    return image, mask

# -------------------------------
# Track entropy / bits generated
# -------------------------------

entropy_history = []

def update_entropy_graph(entropy_value):
    entropy_history.append(entropy_value)

    plt.clf()
    plt.title("Entropy over Time")
    plt.xlabel("Frame Count")
    plt.ylabel("Entropy (bits)")
    plt.plot(entropy_history, marker="o", linestyle="-", color="blue")
    plt.pause(0.01)                    #pause between each frames ie, each bits plotted

#-----------------------------------
# finding the average entropy_value
# ----------------------------------
def avg_entropy(values):
    '''Calculating the average entropy '''

    if not values:
        return 0
    return np.mean(values)
        


#--------------------------------   
# Random basis generator
# -------------------------------
def random_basis(h, w):
    """
    Randomly choose one of the 3 bases (rect/circle/ellipse).
    Place it at a random position and size.
    This simulates "random measurement basis choice" in QM.
    """
    kind = secrets.choice(["rect", "circle", "ellipse"])
    x = secrets.randbelow(w-200) + 100     # avoid edges
    y = secrets.randbelow(h - 200) + 100
    size = secrets.randbelow(51) + 30
    return Basis(kind, (x, y), size)

#--------------------------------
# Shannon Entropy Integration
# -------------------------------

def calculate_shannon_entropy(measurement):
    if measurement.size == 0:
        return 0.0
    counts = np.bincount(measurement, minlength=256)                #this counts the amount of pixels ie the level of randomness
    probabilities = counts[counts > 0] / counts.sum()
    entropy_value = -np.sum(probabilities * np.log2(probabilities)) #here, we use the formula for shannon entropy
    return entropy_value


# -------------------------------
# Collapse measurement into a key
# -------------------------------
def measure_wavefunction(gray, mask):
    measurement = gray[mask == 255]
    if measurement.size == 0:
        return None

    # raw entropy
    data = measurement.tobytes()

    # add system randomness
    salt = os.urandom(16)                                            #16 bytes extra added in order to increase encryption(adds random characters in the external key produced )
    sys_entropy = secrets.token_bytes(16)                            #gets 16 bytes of entropy data from the machine hardware 
    combined = data + salt + sys_entropy 

    # collapse into strong seed
    seed = hashlib.sha512(combined).digest()   # 64 bytes

    # derive final 32-byte key (AES-256 strength)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'webcam-keygen',
        backend=default_backend()
    )
    key = hkdf.derive(seed)

    return key.hex()  # return as hex string for readability

#----------------------------------
#Adds keys to a file 
#----------------------------------
count=0
def add_file(file, key, entropy_value, reset=False):
    """
    Append quantum key + entropy value to CSV.
    """
    global count

    try:
        # Choose append or overwrite
        mode = "w" if reset or not os.path.exists(file) else "a"
        with open(file, mode, newline="") as f:
            writer = csv.writer(f)

            # Write header only if new/reset
            if mode == "w":
                writer.writerow(["ID", "Key Generated", "Entropy Value"])

            count += 1
            writer.writerow([
                count,
                f"Quantum Key: {key[:16]}",
                f"Entropy: {entropy_value:.2f} bits"
            ])
    except Exception as e:
        print(f"[add_file error] {e}")


# --------------------------------
# Main loop
# --------------------------------
def main():
    print("------ Quantum Inspired Key Generator ------")
    print("This application is still under development so~Please look forward to newer updates :D")
    print("======Press ESC to exit, P to pause, S to save screenshot======")


    # Try to open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print(" Cannot access webcam")
        return

    paused = False                # start in running mode
    screenshot_counter = 0        # keep track of screenshots saved

    plt.ion()
    
    while True:
        if not paused:
            # Grab a frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("No webcam feed available")
                break

            # Resize feed → make video bigger
            frame = cv.resize(frame, (960, 720))

            # Flip horizontally → mirror-like display
            frame = cv.flip(frame, 1)

            # Convert to grayscale → pixel brightness values
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Pick a random measurement basis (shape)
            h, w = gray.shape
            basis = random_basis(h, w)

            # Draw shape and get mask
            frame_with_basis, mask = draw_basis_and_mask(frame.copy(), basis)

            # Collapse measurement into key
            key = measure_wavefunction(gray, mask)

            # Compute entropy each frame
            measurement = gray[mask == 255]
            entropy_val = calculate_shannon_entropy(measurement)
            entropy_history.append(entropy_val)
      
            corrected_entropy = entropy_val - 1.05 #subtracting bias entropy 
            update_entropy_graph(corrected_entropy)

            cv.putText(frame_with_basis, f"Entropy: {corrected_entropy:.2f}",
                       (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # If a key was generated, show part of it on screen
            if key:
                cv.putText(frame_with_basis, f"Quantum Key: {key[:16]}",
                           (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

                entropy_val = len(key) * 4  # just a measure (SHA-256 = 256 bits)
                add_file("enc.csv", key, entropy_val)
                

            # Display which basis was used
            cv.putText(frame_with_basis, f"Measurement Basis: {basis.kind}",
                       (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show final output
            cv.imshow("QRNG Simulation", frame_with_basis)

        # -------------------------------
        # Key controls
        # -------------------------------
        keypress = cv.waitKey(300) & 0xFF #we will be producing bulk keys here hence very little wait time
        if keypress == 27:  # ESC = quit
            break
        elif keypress == ord('p'):  # P = pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif keypress == ord('s') and not paused:  # S = save screenshot
            filename = f"screenshot_{int(time.time())}.png"
            cv.imwrite(filename, frame_with_basis)
            print(f"Screenshot saved: {filename}")

    # Cleanup after loop
    cap.release()
    cv.destroyAllWindows()


# -------------------------------
# Run program
# -------------------------------
if __name__ == "__main__":
    main()
