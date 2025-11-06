import matplotlib
# Use TkAgg backend for matplotlib (required for compatibility with OpenCV)
matplotlib.use("TkAgg") 
import cv2 as cv # OpenCV for image processing
import numpy as np # NumPy for numerical operations
import random
from random import choice 
import time  
import csv
import os, secrets, hashlib
import math
import sys
import pandas as pd # Useful for future graphing

# The conflicting CustomTkinter (CTk) and related imports have been removed to prevent crashing.

# --- CRYPTOGRAPHY IMPORTS ---
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
# ----------------------------

# --- CONFIGURATION ---
MINIMUM_ENTROPY_THRESHOLD = 500  # Image variance threshold (adjust if needed)
KEY_LENGTH_BYTES = 32 # 256 bits (Final key length)
MIN_KDF_INPUT_SIZE = 32 # Minimum required bytes for a stable HKDF input

# --- CLASSES ---

class Basis:
    """Represents a randomly selected geometric measurement basis."""
    def __init__(self, kind, mask):
        self.kind = kind
        self.mask = mask

# --- CORE UTILITY FUNCTIONS ---

def calculate_shannon_entropy(data):
    """
    Calculates the Shannon entropy of the key (bytes), safely handling zero-entropy inputs.
    
    This fix prevents the FloatingPointError when data has zero variance (e.g., covered camera).
    """
    if not data:
        return 0
    
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = byte_counts / len(data)
    probabilities = probabilities[probabilities > 0]
    
    # Use np.errstate to suppress floating point warnings (log of zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
    return entropy if not np.isnan(entropy) else 0

def create_random_basis(width, height):
    """Creates a random geometric mask (Basis) to simulate quantum measurement."""
    kind = choice(['rect', 'circle', 'ellipse'])
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x_center = random.randint(width // 4, 3 * width // 4)
    y_center = random.randint(height // 4, 3 * height // 4)
    
    if kind == 'rect':
        x_end = random.randint(x_center + 50, width)
        y_end = random.randint(y_center + 50, height)
        cv.rectangle(mask, (x_center, y_center), (x_end, y_end), 255, -1)
    elif kind == 'circle':
        radius = random.randint(50, min(width, height) // 3)
        cv.circle(mask, (x_center, y_center), radius, 255, -1)
    elif kind == 'ellipse':
        axes = (random.randint(50, width // 4), random.randint(50, height // 4))
        angle = random.randint(0, 180)
        cv.ellipse(mask, (x_center, y_center), axes, angle, 0, 360, 255, -1)
        
    return Basis(kind, mask)

def derive_key(raw_seed: bytes, mode: str = 'TRNG_DEPENDENT') -> bytes:
    """Derives the final cryptographic key using HKDF based on the specified mode."""
    # 1. Determine Salt based on Mode
    if mode == 'TRNG_DEPENDENT':
        # TRNG Mode (Groups A/D): Salt is dependent on the image entropy.
        salt_hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        salt_hasher.update(raw_seed[:KEY_LENGTH_BYTES]) 
        salt = salt_hasher.finalize()
    elif mode == 'CSPRNG_ROBUST':
        # CSPRNG Mode (Group E): Salt is independent (software fallback).
        salt = secrets.token_bytes(KEY_LENGTH_BYTES) 
    else:
        raise ValueError("Invalid KDF mode specified.")

    # 2. Use HKDF to securely mix S_raw with the Salt into a strong key (K)
    kdf = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH_BYTES,
        salt=salt,
        info=b'quantum-inspired-key-derivation',
        backend=default_backend()
    )
    final_key = kdf.derive(raw_seed)
    return final_key

def get_entropy_from_image(image_data: np.ndarray, basis: Basis, kdf_mode: str = 'TRNG_DEPENDENT'):
    """Processes the image data, applies the measurement, and derives the key."""
    # 1. Get image variance
    image_variance = np.var(image_data)
    
    # 2. Apply the randomly selected geometric mask (basis)
    masked_pixels = cv.bitwise_and(image_data, image_data, mask=basis.mask)
    
    # 3. Flatten the masked area to create the raw seed
    raw_seed_array = masked_pixels[basis.mask != 0]
    
    # FIX 2: Stable Raw Seed padding logic
    if raw_seed_array.size == 0:
        raw_seed = b'\x00' * 256
    else:
        raw_seed = raw_seed_array.tobytes()

    # Stable Padding: If the raw_seed is too short, hash it repeatedly
    if len(raw_seed) < MIN_KDF_INPUT_SIZE:
         seed_hash = hashlib.sha256(raw_seed).digest()
         while len(seed_hash) < MIN_KDF_INPUT_SIZE:
             seed_hash += hashlib.sha256(seed_hash).digest()
         raw_seed = seed_hash[:MIN_KDF_INPUT_SIZE] 

    # 4. Derive the key using the specified KDF mode
    final_key = derive_key(raw_seed, kdf_mode)
    
    return final_key, image_variance

def add_file(filename, key, entropy_value, basis_kind, kdf_mode, image_variance):
    """Appends the key data to a CSV file."""
    key_hex = key.hex()
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writerow(["Timestamp", "KDF Mode", "Quantum Key (Hex)", "Image Variance", "Shannon Entropy (Bits)", "Measurement Basis"])
            
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            kdf_mode, 
            key_hex, 
            f"{image_variance:.2f}", 
            f"{entropy_value:.2f} bits", 
            basis_kind
        ])

# --- SIMULATION PARAMETERS ---
KDF_MODE = 'TRNG_DEPENDENT' 
OUTPUT_CSV = 'key_generation_data.csv' 
FRAMES_TO_SKIP = 3  
PROCESSED_KEYS_COUNTER = 0

def run_simulation(cap):
    """The main OpenCV loop for key generation and visualization."""
    global PROCESSED_KEYS_COUNTER
    global KDF_MODE 

    paused = False
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam resolution: {width}x{height}")
    print(f"KDF Mode Set to: {KDF_MODE}")
    print(f"Keys will be logged to: {OUTPUT_CSV}")
    
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam. Exiting.")
            break

        # Frame skipping and throttling logic
        frame_counter = (frame_counter + 1) % FRAMES_TO_SKIP
        if frame_counter != 0 and not paused:
            continue

        if not paused:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            basis = create_random_basis(width, height)
            
            try:
                final_key, image_variance = get_entropy_from_image(gray_frame, basis, KDF_MODE)
                entropy_value = calculate_shannon_entropy(final_key) 

                key_hex = final_key.hex()
                display_text = f"Key: {key_hex[:16]}..."
                
                frame_with_basis = cv.bitwise_and(frame, frame, mask=basis.mask)
                
                # Highlight if entropy is low
                text_color = (0, 255, 0)
                if image_variance < MINIMUM_ENTROPY_THRESHOLD:
                    text_color = (0, 0, 255) # Red for Low Entropy

                # Add text annotations
                cv.putText(frame_with_basis, f"Mode: {KDF_MODE}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.putText(frame_with_basis, f"Basis: {basis.kind}", (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.putText(frame_with_basis, f"Entropy: {entropy_value:.2f} bits", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv.putText(frame_with_basis, f"Variance: {image_variance:.2f}", (20, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv.putText(frame_with_basis, display_text, (20, height - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 5. Log the key
                add_file(OUTPUT_CSV, final_key, entropy_value, basis.kind, KDF_MODE, image_variance)
                PROCESSED_KEYS_COUNTER += 1
                
                cv.putText(frame_with_basis, f"Keys Generated: {PROCESSED_KEYS_COUNTER}", (width - 250, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            except Exception as e:
                # Display error if generation fails
                frame_with_basis = frame.copy()
                cv.putText(frame_with_basis, f"CRASH: {str(e)}", (20, height//2), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                print(f"Generation Error: {e}", file=sys.stderr)


        # If paused, simply display the last frame and a PAUSED message
        else:
            frame_with_basis = frame.copy()
            cv.putText(frame_with_basis, "PAUSED", (width // 2 - 50, height // 2), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


        # Show final output
        cv.imshow("QRNG Simulation", frame_with_basis)

        
        # Key controls 
        keypress = cv.waitKey(1) & 0xFF 

        if keypress == 27:  # ESC = quit
            break
        
        elif keypress == ord('p'):  # P = pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")
        
        elif keypress == ord('m'): # M = Change KDF Mode (TRNG_DEPENDENT <-> CSPRNG_ROBUST)
            if KDF_MODE == 'TRNG_DEPENDENT':
                KDF_MODE = 'CSPRNG_ROBUST'
            else:
                KDF_MODE = 'TRNG_DEPENDENT'
            print(f"KDF Mode switched to: {KDF_MODE}")


    # End of main loop: release webcam
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Initialize the webcam and run the simulation only if successful
    cap = cv.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        run_simulation(cap)