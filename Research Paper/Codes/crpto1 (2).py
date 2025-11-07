import matplotlib
matplotlib.use("TkAgg")
import cv2 as cv
import numpy as np
import random
from random import choice
import time   
import csv
import os, secrets, hashlib
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import matplotlib.pyplot as plt
import customtkinter as CTk
import pandas as pd
import math
from scipy.special import erfc

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

class App(CTk.CTk):

    def __init__(self):
        super().__init__()  
        self.title("crypto")  
        self.geometry("700x600") 
        self.main_frame = CTk.CTkFrame(self,
                                fg_color="#FFDDDD",
                                bg_color="#FFDDDD",
                                border_color="black",  
                                border_width=1)
        self.main_frame.pack(fill="both", expand=True)
        self.header()
        self.button()
        self.button_test()
        
    
    def button(self):
        button = CTk.CTkButton(self.main_frame, 
                            text="Generate Key", 
                            command=main)
        button.pack(pady=20)

    def button_test(self):
        button = CTk.CTkButton(self.main_frame, 
                            text="Do a randomness test", 
                            command=lambda:test_key_randomness("enc.csv"))
        button.pack(pady=30)

    def header(self):
        header = CTk.CTkButton(self.main_frame,
                    text="Quantum Inspired Key Generator",
                    bg_color="#FFDDDD",
                    fg_color="#FFDDDD",
                    hover_color="#FFDDDD",
                    border_color="#FFDDDD",
                    border_width=1,
                    font=CTk.CTkFont(size=40, 
                            weight="bold"),
                    text_color="#000000")
        header.pack(pady=10, padx=0)

# ------------------------------------------
# Basis Class (represents measurement basis)
# ------------------------------------------
class Basis:
    def __init__(self, kind, center, size):
        self.kind = kind      # the type of shape (rect, circle, ellipse)
        self.center = center  # (x, y) center point of the shape
        self.size = size      # size of the shape (radius or half-side length)


# -------------------------------
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
    plt.pause(0.01)

# -------------------------------
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
    counts = np.bincount(measurement, minlength=256) #this counts the amount of pixels ie the level ofrandomness
    probabilities = counts[counts > 0] / counts.sum()
    entropy_value = -np.sum(probabilities * np.log2(probabilities))#here, we use the formula for shannon entropy
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
    salt = os.urandom(16)
    sys_entropy = secrets.token_bytes(16)
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
                f"Quantum Key: {key}",  # *** CORRECTION: REMOVED [:16] SLICE ***
                f"Entropy: {entropy_value:.2f} bits"
            ])
    except Exception as e:
        print(f"[add_file error] {e}")

#-------------------------------
# NIST MONOBIT TEST (NIST TEST-1)
#-------------------------------
def nist_monobit_test(binary_string):
    """
    Perform the NIST SP 800-22 Monobit Frequency Test.
    """
    n = len(binary_string)
    S_n = sum(1 if bit == '1' else -1 for bit in binary_string)
    s_obs = abs(S_n) / math.sqrt(n)
    p_value = erfc(s_obs / math.sqrt(2))
    passed = p_value >= 0.01
    return p_value, passed

#-----------------------------
#Implementing the monobit test
#-----------------------------
def test_key_randomness(file_name):
    """
    Reads all generated keys from the CSV, performs the NIST Monobit Frequency Test
    on each key, and reports the average p-value and pass rate.
    """
    # 1. Initial checks and file reading
    if not os.path.exists(file_name):
        print(f"Error: File '{file_name}' not found.")
        return 0 
    
    try:
        df = pd.read_csv(file_name) 
    except Exception as e:
        print(f"Error reading CSV file. Check formatting or headers: {e}")
        return 0

    if df.empty:
        print("Error: DataFrame is empty. No keys to test.")
        return 0

    # Initialize lists to hold p-values and pass/fail counts
    p_values = []
    pass_count = 0
    fail_count = 0

    for index, row in df.iterrows():
        try:
            # Get the key from the current row
            key_hex_full = row['Key Generated']
            
            # Extract the hex value: splits "Quantum Key: 1a2b..."
            key_hex = key_hex_full.split(':')[1].strip().replace('"', '')
            
            # Convert to 256-bit binary string
            binary = bin(int(key_hex, 16))[2:].zfill(256)
            
            # Perform the NIST Monobit Frequency Test
            p_value, passed = nist_monobit_test(binary)
            p_values.append(p_value)

            if passed:
                pass_count += 1
            else:
                fail_count += 1

        except Exception as e:
            # Catch errors for individual rows (e.g., bad key format)
            print(f"Skipping key at index {index} due to format error: {e}")
            continue # Move to the next key

    # 3. Report Results
    if not p_values:
        print("No valid p-values were calculated.")
        return 0

    average_p_value = sum(p_values) / len(p_values)
    min_p_value = min(p_values)
    pass_rate = pass_count / (pass_count + fail_count) * 100

    print("-" * 40)
    print(f"   Randomness Test Results for {len(p_values)} Keys:")
    print(f"   Average p-value:       **{average_p_value:.4f}**")
    print(f"   Minimum p-value:       **{min_p_value:.4f}**")
    print(f"   Pass Rate:             **{pass_rate:.2f}%**")
    print("-" * 40)

    return average_p_value

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
    
    # Frame skipping control
    FRAMES_TO_SKIP = 5  # Adjust this value as needed
    frame_counter = 0

    # Throttling control
    M = 10  # Adjust this value for throttling frequency
    processed_frames = 0
    
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

        
            
            measurement = gray[mask == 255]
            entropy_value = calculate_shannon_entropy(measurement)
            cv.putText(frame_with_basis, f"Entropy: {entropy_value:.2f} bits", (20, 100),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            update_entropy_graph(entropy_value)

            # If a key was generated, show part of it on screen
            if key:
                cv.putText(frame_with_basis, f"Quantum Key: {key[:16]}",
                        (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

                entropy_val = len(key) * 4  # just a toy measure (SHA-256 = 256 bits)
                add_file("enc.csv", key, entropy_val)

            # Display which basis was used
            cv.putText(frame_with_basis, f"Measurement Basis: {basis.kind}",
                    (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show final output
            cv.imshow("QRNG Simulation", frame_with_basis)

        # -------------------------------
        # Key controls
        # -------------------------------
        keypress = cv.waitKey(33) & 0xFF  # 33ms for ~30 FPS, realistic inter-frame delay

        # Optional: Implement frame skipping to reduce load
        frame_counter = (frame_counter + 1) % FRAMES_TO_SKIP
        if frame_counter != 0:
            continue

        # Optional: Throttle CSV writes and matplotlib updates
        if processed_frames % M == 0:
            # Perform batched writes or updates here
            pass

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

gui = App()
gui.mainloop()
