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
from scipy.special import erfc, gammaincc
from collections import Counter
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
                            command=lambda:test_key_randomness("enc.csv", max_keys=None))
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
    Guarantees shapes stay on-screen with minimum size
    """
    kind = secrets.choice(["rect", "circle", "ellipse"])
    
    # Larger minimum size to guarantee entropy
    size = secrets.randbelow(41) + 80  # 80-120 pixels radius
    
    # Safe boundaries that guarantee full shape visibility
    safe_margin = size + 30
    
    # Ensure we have enough room (fallback to center if screen too small)
    if w < 2 * safe_margin or h < 2 * safe_margin:
        x, y = w // 2, h // 2
    else:
        x = secrets.randbelow(w - 2*safe_margin) + safe_margin
        y = secrets.randbelow(h - 2*safe_margin) + safe_margin
    
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

def xor_fold(data_hex, target_length=64):
    """
    XOR-fold hex string to improve bit distribution.
    Helps with Approximate Entropy test.
    """
    data_bytes = bytes.fromhex(data_hex)
    
    # XOR fold: split in half and XOR together
    mid = len(data_bytes) // 2
    first_half = data_bytes[:mid]
    second_half = data_bytes[mid:mid + len(first_half)]
    
    xored = bytes(a ^ b for a, b in zip(first_half, second_half))
    
    # Mix with remaining bytes if odd length
    if len(data_bytes) % 2:
        xored += data_bytes[-1:] 
    
    return xored.hex()

# -------------------------------
# Collapse measurement into a key
# -------------------------------
# def measure_wavefunction(gray, mask):
#     measurement = gray[mask == 255]
#     if measurement.size == 0:
#         return None

#     # raw entropy
#     data = measurement.tobytes()

#     # add system randomness
#     salt = os.urandom(16)
#     sys_entropy = secrets.token_bytes(16)
#     combined = data + salt + sys_entropy

#     # collapse into strong seed
#     seed = hashlib.sha512(combined).digest()   # 64 bytes

#     # derive final 32-byte key (AES-256 strength)
#     hkdf = HKDF(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=None,
#         info=b'webcam-keygen',
#         backend=default_backend()
#     )
#     key = hkdf.derive(seed)

#     return key.hex()  # return as hex string for readability

def measure_wavefunction(gray, mask):
    measurement = gray[mask == 255]
    
    #  Check minimum pixels
    MIN_PIXELS = 500
    if measurement.size < MIN_PIXELS:
        return None
    
    # Calculate entropy FIRST and validate
    entropy_value = calculate_shannon_entropy(measurement)
    
    # Reject if entropy is too low
    if entropy_value < 2.0:  # Your minimum threshold
        return None
    
    # Rest of your code...
    if len(measurement) > 1:
        xored = np.bitwise_xor(measurement[:-1], measurement[1:])
        measurement = np.append(measurement, xored)
    
    # Von Neumann debiasing on LSBs
    lsbs = (measurement & 1).astype(np.uint8)
    debiased_bits = []
    for i in range(0, len(lsbs)-1, 2):
        if lsbs[i] != lsbs[i+1]:
            debiased_bits.append(lsbs[i])
    
    if len(debiased_bits) > 8:
        debiased_bytes = np.packbits(debiased_bits[:len(debiased_bits)//8*8])
        measurement = np.append(measurement, debiased_bytes)
    
    # Original code starts here
    data = measurement.tobytes()

    #  Multiple entropy sources
    timestamp_entropy = str(time.time_ns()).encode()
    sys_entropy = secrets.token_bytes(32)  # Was 16, now 32
    salt = os.urandom(32)  # Was 16, now 32
    
    # Add pixel variance entropy
    pixel_variance = np.var(measurement).tobytes()
    
    # Add position entropy from mask
    nonzero_positions = np.nonzero(mask)
    position_entropy = hashlib.sha256(
        str(nonzero_positions).encode()
    ).digest()
    
    #Combine all entropy sources
    combined = data + salt + sys_entropy + timestamp_entropy + pixel_variance + position_entropy

    # Use SHA3-512 instead of SHA-512
    seed = hashlib.sha3_512(combined).digest()

    #  Double HKDF derivation for better mixing
    hkdf1 = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b'webcam-keygen-round1',
        backend=default_backend()
    )
    intermediate_key = hkdf1.derive(seed)
    
    hkdf2 = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=sys_entropy[:16],
        info=b'webcam-keygen-round2',
        backend=default_backend()
    )
    key = hkdf2.derive(intermediate_key)

    #  Apply XOR folding (see helper function below)
    key_hex = key.hex()
    final_key = xor_fold(key_hex)
    
    return final_key

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
                f"Quantum Key: {key}",  
                f"Entropy: {entropy_value:.2f} bits"
            ])
    except Exception as e:
        print(f"[add_file error] {e}")

def nist_monobit_test(binary_string):
    """
    Test 1: Monobit Frequency Test
    Checks if the number of 0s and 1s are roughly equal.
    
    How it works:
    - Convert bits to +1 (for '1') and -1 (for '0')
    - Sum them up
    - Random data should sum close to 0
    """
    n = len(binary_string)
    
    # Convert to +1/-1 and sum
    S_n = sum(1 if bit == '1' else -1 for bit in binary_string)
    
    # Calculate test statistic
    s_obs = abs(S_n) / math.sqrt(n)
    
    # Calculate p-value using complementary error function
    p_value = erfc(s_obs / math.sqrt(2))
    
    passed = p_value >= 0.01
    return p_value, passed


def nist_runs_test(binary_string):
    """
    Test 2: Runs Test
    Checks if the number of runs (consecutive identical bits) is appropriate.
    
    How it works:
    - Count transitions from 0->1 or 1->0
    - Too few runs = bits clump together
    - Too many runs = bits alternate too perfectly
    """
    n = len(binary_string)
    
    # Pre-requisite: proportion of 1s should be close to 0.5
    pi = binary_string.count('1') / n
    
    # If pre-test fails, return 0
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0, False
    
    # Count runs (a run is a sequence of identical bits)
    runs = 1  # Start with 1 run
    for i in range(1, n):
        if binary_string[i] != binary_string[i-1]:
            runs += 1
    
    # Calculate test statistic
    numerator = abs(runs - 2 * n * pi * (1 - pi))
    denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    
    if denominator == 0:
        return 0.0, False
    
    # Calculate p-value
    p_value = erfc(numerator / denominator)
    
    passed = p_value >= 0.01
    return p_value, passed


def nist_longest_run_test(binary_string):
    """
    Test 3: Longest Run of Ones Test
    Checks if the longest runs of 1s are as expected.
    
    How it works:
    - Divide sequence into blocks
    - Find longest run of 1s in each block
    - Compare distribution to expected values
    """
    n = len(binary_string)
    
    # Determine parameters based on sequence length
    if n < 128:
        return 0.0, False  # Too short
    elif n < 6272:
        M = 8
        K = 3
        N = 16
        v_values = [1, 2, 3, 4]
        pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M = 128
        K = 5
        N = 49
        v_values = [4, 5, 6, 7, 8, 9]
        pi_values = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M = 10000
        K = 6
        N = 75
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pi_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    
    # Divide into blocks
    blocks = [binary_string[i:i+M] for i in range(0, N*M, M)]
    
    # Count longest runs in each block
    v_counts = [0] * len(v_values)
    
    for block in blocks:
        # Find longest run of 1s
        max_run = 0
        current_run = 0
        
        for bit in block:
            if bit == '1':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        # Categorize this run length
        for i, v in enumerate(v_values):
            if i == 0 and max_run <= v:
                v_counts[i] += 1
                break
            elif i == len(v_values) - 1 and max_run >= v:
                v_counts[i] += 1
                break
            elif i > 0 and v_values[i-1] < max_run <= v:
                v_counts[i] += 1
                break
    
    # Calculate chi-square statistic
    chi_square = sum(
        (v_counts[i] - N * pi_values[i])**2 / (N * pi_values[i])
        for i in range(len(v_values))
    )
    
    # Calculate p-value using incomplete gamma function
    p_value = gammaincc(K / 2, chi_square / 2)
    
    passed = p_value >= 0.01
    return p_value, passed


def nist_spectral_test(binary_string):
    """
    Test 4: Discrete Fourier Transform (Spectral) Test
    Checks for periodic patterns using frequency analysis.
    
    How it works:
    - Convert bits to +1/-1
    - Apply FFT to find frequency components
    - Random data should have flat frequency spectrum (white noise)
    - Peaks indicate periodic patterns
    """
    n = len(binary_string)
    
    # Convert to +1/-1
    X = np.array([1 if bit == '1' else -1 for bit in binary_string])
    
    # Apply FFT (only need first half due to symmetry)
    S = np.fft.fft(X)
    M = abs(S[:n//2])
    
    # Calculate threshold (95% of values should be below this)
    T = math.sqrt(math.log(1/0.05) * n)
    
    # Count peaks above threshold
    N0 = 0.95 * n / 2
    N1 = len(M[M < T])
    
    # Calculate test statistic
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    
    # Calculate p-value
    p_value = erfc(abs(d) / math.sqrt(2))
    
    passed = p_value >= 0.01
    return p_value, passed


def nist_approximate_entropy_test(binary_string, m=2):
    """
    Test 5: Approximate Entropy Test
    Checks if patterns of length m and m+1 appear with expected frequency.
    
    How it works:
    - Count all overlapping m-bit patterns
    - Count all overlapping (m+1)-bit patterns
    - Compare their frequency distributions
    - Random data should have unpredictable pattern frequencies
    """
    n = len(binary_string)
    
    def phi(m_val):
        """Calculate phi for pattern length m"""
        # Count all overlapping patterns of length m
        patterns = {}
        for i in range(n):
            pattern = binary_string[i:i+m_val]
            if len(pattern) == m_val:
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Calculate sum of log probabilities
        phi_sum = 0
        for count in patterns.values():
            if count > 0:
                p = count / n
                phi_sum += p * math.log(p)
        
        return phi_sum
    
    # Calculate phi for m and m+1
    phi_m = phi(m)
    phi_m_plus_1 = phi(m + 1)
    
    # Calculate ApEn (Approximate Entropy)
    apen = phi_m - phi_m_plus_1
    
    # Calculate chi-square statistic
    chi_square = 2 * n * (math.log(2) - apen)
    
    # Calculate p-value
    p_value = gammaincc(2**(m-1), chi_square / 2)
    
    passed = p_value >= 0.01
    return p_value, passed


# ============================================================
# BATCH TESTING FUNCTION
# ============================================================

def run_all_nist_tests(binary_string, verbose=False):
    """
    Run all 5 NIST tests and return comprehensive results.
    
    Args:
        binary_string: Binary string to test
        verbose: If True, print detailed results for each test
    
    Returns:
        dict: Test names mapped to (p_value, passed) tuples
    """
    tests = {
        "Monobit": nist_monobit_test,
        "Runs": nist_runs_test,
        "Longest Run": nist_longest_run_test,
        "Spectral": nist_spectral_test,
        "Approximate Entropy": nist_approximate_entropy_test
    }
    
    results = {}
    
    if verbose:
        print("\n" + "="*60)
        print("NIST SP 800-22 Randomness Test Suite")
        print("="*60)
        print(f"Testing {len(binary_string)} bits...")
        print(f"Pass threshold: p-value >= 0.01\n")
    
    for test_name, test_func in tests.items():
        try:
            p_value, passed = test_func(binary_string)
            results[test_name] = (p_value, passed)
            
            if verbose:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{test_name:.<30} p={p_value:.6f} ... {status}")
        except Exception as e:
            if verbose:
                print(f"{test_name:.<30} ERROR: {e}")
            results[test_name] = (0.0, False)
    
    if verbose:
        passed_count = sum(1 for _, passed in results.values() if passed)
        total_count = len(results)
        print("\n" + "-"*60)
        print(f"SUMMARY: {passed_count}/{total_count} tests passed")
        print("="*60 + "\n")
    
    return results

#-----------------------------
#Implementing the monobit test
#-----------------------------
def test_key_randomness(file_name, max_keys=None):
    """
    Reads all generated keys from the CSV and runs the full NIST test suite.
    
    Args:
        file_name: Path to CSV file with keys
        max_keys: Maximum number of keys to test (None = test all keys)
    """
    print("\n[Starting randomness tests...]")
    
    if not os.path.exists(file_name):
        print(f"Error: File '{file_name}' not found.")
        return
    
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("Error: No keys to test.")
        return

    # Test ALL keys if max_keys is None, otherwise limit
    num_keys = len(df) if max_keys is None else min(len(df), max_keys)
    print(f"Testing {num_keys} keys from {file_name}...\n")

    # Store all test results
    all_results = {
        "Monobit": [], 
        "Runs": [], 
        "Longest Run": [], 
        "Spectral": [], 
        "Approximate Entropy": []
    }

    # Test each key (with progress indicator)
    for index in range(num_keys):
        try:
            row = df.iloc[index]
            key_hex_full = row['Key Generated']
            key_hex = key_hex_full.split(':')[1].strip().replace('"', '')
            binary = bin(int(key_hex, 16))[2:].zfill(256)
            
            # Run all 5 tests (verbose=False for speed)
            results = run_all_nist_tests(binary, verbose=False)
            
            # Store results
            for test_name, (p_value, passed) in results.items():
                all_results[test_name].append((p_value, passed))
            
            # Progress indicator every 100 keys
            if (index + 1) % 100 == 0:
                print(f"  Processed {index + 1}/{num_keys} keys...")
                
        except Exception as e:
            print(f"Skipping key {index}: {e}")
            continue

    # Print summary statistics
    print("\n" + "="*60)
    print("AGGREGATE RESULTS ACROSS ALL KEYS")
    print("="*60)
    
    for test_name, results in all_results.items():
        if results:
            p_values = [p for p, _ in results]
            passes = sum(1 for _, passed in results if passed)
            
            avg_p = sum(p_values) / len(p_values)
            min_p = min(p_values)
            max_p = max(p_values)
            pass_rate = (passes / len(results)) * 100
            
            print(f"\n{test_name}:")
            print(f"  Average p-value: {avg_p:.4f}")
            print(f"  Min p-value:     {min_p:.4f}")
            print(f"  Max p-value:     {max_p:.4f}")
            print(f"  Pass rate:       {pass_rate:.1f}% ({passes}/{len(results)})")
    
    print("\n" + "="*60)
    print(f"[Testing complete! Tested {num_keys} keys]")
    print("="*60 + "\n")
    
    return all_results  # Return results for further analysis
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
    FRAMES_TO_SKIP = 10 # Adjust this value as needed
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

            # Pick a random measurement basis
            h, w = gray.shape
            basis = random_basis(h, w)

            # Draw shape and get mask
            frame_with_basis, mask = draw_basis_and_mask(frame.copy(), basis)

            #  NEW: Check pixel count BEFORE attempting key generation
            measurement = gray[mask == 255]
            pixel_count = measurement.size

# NEW: Retry if insufficient pixels (up to 5 attempts)
            retry_count = 0
            while pixel_count < 500 and retry_count < 5:
                basis = random_basis(h, w)
                frame_with_basis, mask = draw_basis_and_mask(frame.copy(), basis)
                measurement = gray[mask == 255]
                pixel_count = measurement.size
                retry_count += 1

            #  NEW: Skip this frame entirely if still insufficient
            if pixel_count < 500:
                cv.putText(frame_with_basis, "Insufficient entropy - skipping frame", 
                        (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv.imshow("QRNG Simulation", frame_with_basis)
                continue  # Skip to next frame

            # Now we're guaranteed to have enough pixels
            entropy_value = calculate_shannon_entropy(measurement)
            key = measure_wavefunction(gray, mask)

                    
            
            measurement = gray[mask == 255]
            entropy_value = calculate_shannon_entropy(measurement)
            cv.putText(frame_with_basis, f"Entropy: {entropy_value:.2f} bits", (20, 100),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            update_entropy_graph(entropy_value)

            # If a key was generated, show part of it on screen
            # if key:
            #     cv.putText(frame_with_basis, f"Quantum Key: {key[:16]}",
            #             (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

            #     entropy_val = len(key) * 4  # just a toy measure (SHA-256 = 256 bits)
            #     add_file("enc.csv", key, entropy_val)

            # # Display which basis was used
            # cv.putText(frame_with_basis, f"Measurement Basis: {basis.kind}",
            #         (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if key and entropy_value >= 2.0:
                cv.putText(frame_with_basis, f"Quantum Key: {key[:16]}",
                        (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

                add_file("enc.csv", key, entropy_value, basis.kind)
                update_entropy_graph(entropy_value)  # Only plot valid entropy
            elif key is None:
                cv.putText(frame_with_basis, "Low entropy - skipped",
                        (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
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
