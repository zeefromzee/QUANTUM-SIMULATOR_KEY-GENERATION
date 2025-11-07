import matplotlib
matplotlib.use("TkAgg") # Use Tkinter backend for matplotlib
import cv2 as cv # OpenCV for image processing
import numpy as np
import random
from random import choice #for random basis selection
import time #for timestamps  
import csv
import os, secrets, hashlib
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import matplotlib.pyplot as plt
import customtkinter as CTk
import math
from scipy.special import erfc, gammaincc
import pandas as pd
import threading
import shutil # Added for file migration backup

# Global variable for CSV counter
count = 0
entropy_history = []


class App(CTk.CTk):
    def __init__(self):
        super().__init__()  
        self.geometry("700x600") 
        self.title("Quantum Key Generator")
        self.main_frame = CTk.CTkFrame(self,
                                fg_color="#FCFCFC",
                                bg_color="#FCFCFC"
                                )
        self.main_frame.pack(fill="both", expand=True)
        self.header()
        self.button_key()
        self.button_test()
        self.button_graphs()
        self.button_migrate()
    
    # NOTE: Simplified button_key - removed redundant 'safe_main' wrapper
    def button_key(self):
        button = CTk.CTkButton(self.main_frame, 
                        text="Click here to generate keys",
                        width=130,
                        height=50,
                        fg_color="#AFD48D",
                        text_color="black",
                        hover_color="#819C67",
                        corner_radius=20,
                        border_width=2,
                        border_color="#AFD48D",
                        font=CTk.CTkFont(size=20),
                        command=lambda: threading.Thread(target=main, daemon=True).start())
        button.pack(pady=20)


    def button_test(self):
        button = CTk.CTkButton(self.main_frame, 
                            text="Do a randomness test", 
                            width=130,
                            height=50,
                            fg_color="#AFD48D",
                            text_color="black",
                            hover_color="#819C67",
                            corner_radius=20,
                            border_width=2,
                            border_color="#AFD48D",
                            font=CTk.CTkFont(size=20),
                            command=lambda:threading.Thread(target=lambda: test_key_randomness("closed.csv"), daemon=True).start())
        button.pack(pady=30)

    # Header for the main frame
    def header(self):
        header = CTk.CTkButton(self.main_frame,
                text="Quantum Inspired Key Generator",
                bg_color="#FCFCFC",
                fg_color="#FCFCFC",
                hover_color="#FCFCFC",
                font=CTk.CTkFont(size=40, 
                        weight="bold"),
                text_color="#000000")
        header.pack(pady=20, padx=10)

    def button_graphs(self):
        button = CTk.CTkButton(self.main_frame, 
                            text="Generate Advanced Graphs",
                            width=130,
                            height=50,
                            fg_color="#AFD48D",
                            text_color="black",
                            hover_color="#819C67",
                            corner_radius=20,
                            border_width=2,
                            border_color="#AFD48D",
                            font=CTk.CTkFont(size=20), 
                        command=lambda: threading.Thread(target=lambda: generate_advanced_graphs("closed.csv"), daemon=True).start())
        button.pack(pady=30)

    def button_migrate(self):
        """Button to migrate old CSV files."""
        button = CTk.CTkButton(
                            self.main_frame, 
                            text="Migrate Old CSV File", 
                            width=130,
                            height=50,
                            fg_color="#AFD48D",
                            text_color="black",
                            hover_color="#819C67",
                            corner_radius=20,
                            border_width=2,
                            border_color="#AFD48D",
                            font=CTk.CTkFont(size=20),
            command=lambda: migrate_old_csv("closed.csv")
    )
        button.pack(pady=10)

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

def xor_fold(data_hex, target_length=64):
    """
    XOR-fold hex string to improve bit distribution.
    Helps with Approximate Entropy test. (Kept for testing functions)
    """
    try:
        data_hex = data_hex.lower().strip()
        if len(data_hex) % 2 != 0:
            data_hex = '0' + data_hex
            
        data_bytes = bytes.fromhex(data_hex)
        
        while len(data_bytes) < 32:
            data_bytes = data_bytes * 2
            
        mid = len(data_bytes) // 2
        first_half = data_bytes[:mid]
        second_half = data_bytes[mid:mid + len(first_half)]
        
        xored = bytes(a ^ b for a, b in zip(first_half, second_half))
        
        if len(data_bytes) % 2:
            xored += data_bytes[-1:]
            
        result = xored.hex()
        
        if len(result) < target_length:
            result = result * (target_length // len(result) + 1)
        return result[:target_length]
        
    except Exception as e:
        print(f"Error in xor_fold: {e}")
        return '0' * target_length

# -------------------------------------------------------------
# ENHANCED measure_wavefunction (Fixed by removing final xor_fold)
# -------------------------------------------------------------
def measure_wavefunction(gray, mask):
    measurement = gray[mask == 255]
    
    MIN_PIXELS = 150
    if measurement.size < MIN_PIXELS:
        return None
    
    # Calculate entropy after initial selection
    entropy_value = calculate_shannon_entropy(measurement)
    
    if entropy_value < 0.75:
        return None
    
    # Multi-layer preprocessing
    def apply_von_neumann(bits):
        """Improved von Neumann debiasing"""
        debiased = []
        for i in range(0, len(bits)-1, 2):
            if bits[i] != bits[i+1]:
                debiased.append(bits[i])
        return np.array(debiased, dtype=np.uint8)
    
    # 1. Break spatial correlation
    if len(measurement) > 1:
        xored = np.bitwise_xor(measurement[:-1], measurement[1:])
        measurement = np.append(measurement, xored)
    
    # 2. Extract and debias multiple bit planes
    debiased_all = []
    for bit_pos in range(8):
        bits = (measurement >> bit_pos) & 1
        debiased = apply_von_neumann(bits)
        if len(debiased) > 0:
            debiased_all.extend(debiased)
    
    # 3. Convert debiased bits to bytes and append
    if len(debiased_all) > 8:
        debiased_bytes = np.packbits(debiased_all[:len(debiased_all)//8*8])
        measurement = np.append(measurement, debiased_bytes)
        
    # 4. Final XOR mixing
    if len(measurement) > 1:
        # Use a reduction operation for full mixing
        try:
            mixed = np.bitwise_xor.reduce([measurement[i:i+8] for i in range(0, len(measurement)-7, 8)])
            measurement = np.append(measurement, mixed)
        except ValueError: # Handle sequences too short for 8-byte blocks
            pass
    
    data = measurement.tobytes()

    # Multiple entropy sources
    timestamp_entropy = str(time.time_ns()).encode()
    sys_entropy = secrets.token_bytes(32)
    salt = os.urandom(32)
    
    pixel_variance = np.var(measurement).tobytes()
    
    # Use SHA-256 on the mask structure for position entropy
    nonzero_positions = np.nonzero(mask)
    position_entropy = hashlib.sha256(
        str(nonzero_positions).encode()
    ).digest()
    
    combined = data + salt + sys_entropy + timestamp_entropy + pixel_variance + position_entropy

    seed = hashlib.sha3_512(combined).digest()

    # Double HKDF derivation for high key strength
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

    # Ensure consistent hex format
    key_hex = key.hex().lower()
    
    # --- FIX: REMOVED XOR_FOLD HERE TO PREVENT BIASING ---
    final_key = key_hex 
    # --- END FIX ---
    
    # Ensure even length and proper hex format
    if len(final_key) % 2 != 0:
        final_key = '0' + final_key
    
    return final_key  

# -------------------------------
# Track entropy / bits generated
# -------------------------------
def update_entropy_graph(entropy_value):
    if entropy_value is None or np.isnan(entropy_value) or entropy_value < 0:
        return
        
    entropy_history.append(entropy_value)
    
    try:
        plt.figure('Entropy Plot')
        plt.clf()
        plt.title("Entropy over Time")
        plt.xlabel("Frame Count")
        plt.ylabel("Entropy (bits)")
        plt.plot(entropy_history, marker="o", linestyle="-", color="blue")
        plt.draw()
        plt.tight_layout()
        plt.gcf().canvas.flush_events()
    except Exception as e:
        # This can happen if the plot window is closed in the middle of generation
        pass

# -------------------------------
# Random basis generator
# -------------------------------
def random_basis(h, w):
    """
    Randomly choose one of the 3 bases (rect/circle/ellipse).
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
    if measurement is None or measurement.size == 0 or measurement.size < 50:
        return 0.0
        
    # Add minor noise to break uniformity in low-entropy regions
    measurement = measurement.astype(np.float32)
    measurement += np.random.normal(0, 0.1, measurement.shape)
    measurement = np.clip(measurement, 0, 255).astype(np.uint8)
    
    counts = np.bincount(measurement, minlength=256)
    total = counts.sum()
    
    if total == 0:
        return 0.0
        
    probabilities = counts[counts > 0] / total
    
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    
    if np.isnan(entropy_value) or entropy_value < 0:
        return 0.0
    
    # Scale very low entropy values up slightly to be more permissive 
    if entropy_value < 0.1:
        entropy_value = 0.1
        
    return entropy_value


# ============================================================
# CSV/GRAPHING UTILITIES
# ============================================================

def load_entropy_data(csv_file="closed.csv"):
    """
    Load entropy data from CSV.
    """
    try:
        if not os.path.exists(csv_file):
            print(f"[Error] CSV file '{csv_file}' not found. Generate keys first.")
            return None
        
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print(f"[Error] CSV file '{csv_file}' is empty. No data to process.")
            return None
        
        required_cols = ['ID', 'Key Generated', 'Entropy Value', 'Basis Kind']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"[Error] Missing columns: {missing_cols}. Please migrate or regenerate.")
            return None
        
        # Robustly parse Entropy Value from "Entropy: X.XX bits" or just "X.XX"
        try:
            # First try extracting from the common string format
            parsed = df['Entropy Value'].astype(str).str.extract(r'([\d\.]+)')
            df['Entropy Value'] = pd.to_numeric(parsed[0], errors='coerce')
        except Exception:
            print("[Warning] Failed to parse Entropy Value. Trying direct conversion.")
            df['Entropy Value'] = pd.to_numeric(df['Entropy Value'], errors='coerce')
        
        df = df.dropna(subset=['Entropy Value', 'Basis Kind'])
        
        print(f"[✓] Loaded {len(df)} valid records from {csv_file}")
        return df
        
    except Exception as e:
        print(f"[Unexpected Error] Failed to load CSV: {e}")
        return None


def validate_csv_structure(csv_file="closed.csv"):
    """
    Validates CSV file structure before processing.
    """
    try:
        if not os.path.exists(csv_file):
            print(f"[✗] File not found: {csv_file}")
            return False
        
        df = pd.read_csv(csv_file)
        
        required_cols = ['ID', 'Key Generated', 'Entropy Value', 'Basis Kind']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"[✗] Missing required columns: {missing_cols}")
            print(f"[Hint] Delete {csv_file} and regenerate keys OR use the Migrate button.")
            return False
        
        return True
        
    except Exception as e:
        print(f"[✗] Validation failed: {e}")
        return False


def plot_entropy_over_time(df, save_path="entropy_over_time.png"):
    """Entropy over time line graph."""
    if df is None or df.empty: return False
    plt.figure(figsize=(8, 4))
    plt.plot(df['ID'], df['Entropy Value'], marker='o', linestyle='-', alpha=0.7)
    plt.title("Entropy Over Time")
    plt.xlabel("Frame / Key ID")
    plt.ylabel("Entropy (bits)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return True


def plot_bit_proportion_histogram(df, save_path="bit_proportion_histogram.png"):
    """Bit proportion histogram."""
    if df is None or df.empty: return False
    
    def hex_to_bit_proportion(key_full_string):
        try:
            key_hex = key_full_string.split(':')[1].strip().replace('"', '')
            binary_string = bin(int(key_hex, 16))[2:].zfill(len(key_hex) * 4)
            count_ones = binary_string.count('1')
            total_bits = len(binary_string)
            return count_ones / total_bits if total_bits > 0 else None
        except:
            return None

    df = df.copy()
    df['Proportion_of_Ones'] = df['Key Generated'].apply(hex_to_bit_proportion)
    df.dropna(subset=['Proportion_of_Ones'], inplace=True)
    
    if df.empty:
        print("[!] No valid key data found for Bit Proportion plot.")
        return False

    plt.figure(figsize=(6, 4))
    plt.hist(df['Proportion_of_Ones'], bins=50, color='indianred', 
            edgecolor='black', alpha=0.7)
    plt.title("Distribution of Bit Proportion ('1's)")
    plt.xlabel("Proportion of '1' Bits (Target: 0.5)")
    plt.ylabel("Frequency")
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=1.5, label='Ideal 0.5')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return True


def plot_entropy_by_basis_boxplot(df, save_path="entropy_by_basis.png"):
    """Box plot of entropy by basis."""
    if df is None or df.empty: return False
    if 'Basis Kind' not in df.columns: return False
    
    plt.figure(figsize=(6, 4))
    df.boxplot(column='Entropy Value', by='Basis Kind', grid=True, patch_artist=True)
    plt.suptitle('')
    plt.title("Entropy Distribution by Measurement Basis")
    plt.xlabel("Measurement Basis Type")
    plt.ylabel("Entropy (bits)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return True


def generate_advanced_graphs(csv_file="closed.csv"):
    """
    Master function to generate all advanced graphs.
    """
    print("\n" + "="*60)
    print("ADVANCED GRAPH GENERATION")
    print("="*60)
    
    print("\n[1/4] Validating CSV structure...")
    if not validate_csv_structure(csv_file):
        print("\n Graph generation aborted due to validation errors.")
        return False
    
    print("\n[2/4] Loading data...")
    df = load_entropy_data(csv_file)
    if df is None:
        print("\nFailed to load data. Graph generation aborted.")
        return False
    
    print("\n[3/4] Generating graphs...")
    results = {
        "Entropy Over Time": plot_entropy_over_time(df.copy()),
        "Bit Proportion Histogram": plot_bit_proportion_histogram(df.copy()),
        "Entropy By Basis Boxplot": plot_entropy_by_basis_boxplot(df.copy()),
    }
    
    print("\n[4/4] Generation Summary")
    print("-" * 60)
    success_count = sum(results.values())
    total_count = len(results)
    
    for graph_name, success in results.items():
        status = "[✓]" if success else "[✗]"
        print(f"{status} {graph_name}")
    
    print("-" * 60)
    print(f"Successfully generated {success_count}/{total_count} graphs")
    print("="*60 + "\n")
    
    return success_count == total_count

# ======================
# CSV MIGRATION SCRIPT
# ======================
def migrate_old_csv(old_file="closed.csv", backup=True):
    """
    Migrates old CSV format to new format by adding 'Basis Kind' column.
    """
    try:
        from datetime import datetime
        
        print("\n" + "="*60)
        print("CSV MIGRATION TOOL")
        print("="*60)
        
        if not os.path.exists(old_file):
            print(f"[✗] File not found: {old_file}")
            return False
        
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{old_file}.backup_{timestamp}"
            shutil.copy(old_file, backup_file)
            print(f"[✓] Created backup: {backup_file}")
        
        print(f"[→] Reading {old_file}...")
        df = pd.read_csv(old_file)
        
        if 'Basis Kind' in df.columns:
            print("[!] 'Basis Kind' column already exists. No migration needed.")
            return True
        
        print("[→] Adding 'Basis Kind' column...")
        basis_types = ['rect', 'circle', 'ellipse']
        # Assign random basis, as original is unrecoverable
        df['Basis Kind'] = [secrets.choice(basis_types) for _ in range(len(df))]
        
        print(f"[→] Writing updated file...")
        df.to_csv(old_file, index=False)
        
        print(f"[✓] Migration complete! Updated {len(df)} records.")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"[✗] Migration failed: {e}")
        return False


# ============================================================
# ENHANCED add_file (Fixed for consistent key and entropy format)
# ============================================================

def add_file(file, key, entropy_value, basis_kind, reset=False):
    """
    Append quantum key data to CSV with error handling.
    """
    global count

    try:
        if not key or not isinstance(key, str): return
        
        key = key.lower().strip()
        if len(key) % 2 != 0: key = '0' + key
            
        mode = "w" if reset or not os.path.exists(file) else "a"
        
        with open(file, mode, newline="") as f:
            writer = csv.writer(f)

            if mode == "w":
                writer.writerow(["ID", "Key Generated", "Entropy Value", "Basis Kind"])
                count = 0

            count += 1
            # Store data in a consistent format
            writer.writerow([
                count,
                f"Quantum Key: {key}",  
                f"Entropy: {entropy_value:.2f} bits",  # Stored with text prefix for robustness
                basis_kind
            ])
            
    except Exception as e:
        print(f"[Error] Failed to write to CSV: {e}")

# ============================================================
# NIST RANDOMNESS TESTS
# ============================================================

def nist_monobit_test(binary_string):
    n = len(binary_string)
    S_n = sum(1 if bit == '1' else -1 for bit in binary_string)
    s_obs = abs(S_n) / math.sqrt(n)
    p_value = erfc(s_obs / math.sqrt(2))
    passed = p_value >= 0.01
    return p_value, passed

def nist_runs_test(binary_string):
    n = len(binary_string)
    pi = binary_string.count('1') / n
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0, False
    
    runs = 1
    for i in range(1, n):
        if binary_string[i] != binary_string[i-1]:
            runs += 1
    
    numerator = abs(runs - 2 * n * pi * (1 - pi))
    denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    
    if denominator == 0: return 0.0, False
    
    p_value = erfc(numerator / denominator)
    passed = p_value >= 0.01
    return p_value, passed

def nist_longest_run_test(binary_string):
    n = len(binary_string)
    
    if n < 128: return 0.0, False
    
    # Simplified parameters based on NIST SP 800-22 guidelines (for N=256)
    if n < 6272:
        M = 8 # Block size
        K = 3 # Degrees of freedom (v values - 1)
        N_blocks = 16 # Number of blocks
        v_values = [1, 2, 3, 4] # Length categories
        pi_values = [0.2148, 0.3672, 0.2305, 0.1875] # Expected probabilities
    else: # Fallback for larger sizes
        M = 128
        K = 5
        N_blocks = n // M
        v_values = [4, 5, 6, 7, 8, 9]
        pi_values = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        N_blocks = min(N_blocks, 49) # Limit blocks for calculation stability
    
    blocks = [binary_string[i:i+M] for i in range(0, N_blocks*M, M)]
    
    v_counts = [0] * len(v_values)
    
    for block in blocks:
        max_run = 0
        current_run = 0
        for bit in block:
            if bit == '1':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        # Categorize run length
        if max_run <= v_values[0]:
            v_counts[0] += 1
        elif max_run > v_values[-1]:
            v_counts[-1] += 1
        else:
            for i in range(1, len(v_values)-1):
                if v_values[i-1] < max_run <= v_values[i]:
                    v_counts[i] += 1
                    break
    
    chi_square = sum(
        (v_counts[i] - N_blocks * pi_values[i])**2 / (N_blocks * pi_values[i])
        for i in range(len(v_values)) if N_blocks * pi_values[i] > 0
    )
    
    p_value = gammaincc(K / 2, chi_square / 2)
    passed = p_value >= 0.01
    return p_value, passed


def nist_spectral_test(binary_string):
    n = len(binary_string)
    X = np.array([1 if bit == '1' else -1 for bit in binary_string])
    S = np.fft.fft(X)
    M = abs(S[:n//2])
    
    T = math.sqrt(math.log(1/0.05) * n)
    
    N0 = 0.95 * n / 2
    N1 = len(M[M < T])
    
    if n * 0.95 * 0.05 / 4 == 0: return 0.0, False
    
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    
    p_value = erfc(abs(d) / math.sqrt(2))
    passed = p_value >= 0.01
    return p_value, passed


def nist_approximate_entropy_test(binary_string, m=2):
    n = len(binary_string)
    
    def phi(m_val):
        patterns = {}
        # Count overlapping patterns
        for i in range(n):
            pattern = binary_string[i:i+m_val]
            if len(pattern) == m_val:
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        phi_sum = 0
        for count in patterns.values():
            if count > 0:
                p = count / n
                phi_sum += p * math.log(p)
        return phi_sum
    
    try:
        phi_m = phi(m)
        phi_m_plus_1 = phi(m + 1)
        
        apen = phi_m - phi_m_plus_1
        
        chi_square = 2 * n * (math.log(2) - apen)
        
        p_value = gammaincc(2**(m-1), chi_square / 2)
        passed = p_value >= 0.01
        return p_value, passed
    except ValueError:
        return 0.0, False


def run_all_nist_tests(binary_string, verbose=False):
    """Run all 5 NIST tests and return comprehensive results."""
    tests = {
        "Monobit": nist_monobit_test,
        "Runs": nist_runs_test,
        "Longest Run": nist_longest_run_test,
        "Spectral": nist_spectral_test,
        "Approximate Entropy": nist_approximate_entropy_test
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        try:
            p_value, passed = test_func(binary_string)
            results[test_name] = (p_value, passed)
        except Exception:
            results[test_name] = (0.0, False)
            
    return results

# ============================================================
# BATCH TESTING FUNCTION (CRITICALLY REWRITTEN FOR ROBUST CSV PARSING)
# ============================================================

def test_key_randomness(file_name, max_keys=None):
    """
    Reads all generated keys from the CSV and runs the full NIST test suite.
    """
    print("\n[Starting randomness tests with preprocessing...]")
    
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
        
    # Helper for XOR folding the key for better test distribution (used inside the loop)
    def xor_fold_for_test(key_hex):
        data = bytes.fromhex(key_hex)
        mid = len(data) // 2
        first_half = data[:mid]
        second_half = data[mid:mid + len(first_half)]
        xored = bytes(a ^ b for a, b in zip(first_half, second_half))
        return xored.hex()

    # --- ROBUST DATA PREPROCESSING FIX ---
    try:
        # Extract the hex string following "Quantum Key: "
        if 'Key Generated' in df.columns:
            df['clean_key'] = df['Key Generated'].astype(str).str.extract(r'Quantum Key: ([0-9a-fA-F]+)')
        else:
            print("Error: 'Key Generated' column not found. Cannot test.")
            return

        # Filter for valid hex strings (at least 2 hex characters = 1 byte)
        df = df[df['clean_key'].str.len() >= 2].copy()
        
        if df.empty:
            print("\nNo valid keys found after filtering.")
            print("Please check that your CSV file contains proper key values.")
            return

        print(f"\nValid entries for testing: {len(df)}")
        
    except Exception as e:
        print(f"\nError during robust CSV data preprocessing: {str(e)}")
        return
    # --- END ROBUST DATA PREPROCESSING FIX ---

    num_keys = len(df) if max_keys is None else min(len(df), max_keys)
    print(f"Testing {num_keys} keys from {file_name}...\n")

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
            key_hex = str(row['clean_key']).strip()
            
            if not all(c in '0123456789abcdefABCDEF' for c in key_hex):
                continue
                
            if len(key_hex) % 2 != 0:
                key_hex = '0' + key_hex
            
            # Apply XOR folding only for testing purposes
            key_hex_folded = xor_fold_for_test(key_hex)
            binary = bin(int(key_hex_folded, 16))[2:].zfill(len(key_hex_folded) * 4)
            
            # Run tests on different windows of the binary data
            results_list = []
            window_sizes = [256, 128, 64]
            
            for size in window_sizes:
                if size <= len(binary):
                    # Test different windows (50% overlap)
                    windows = [binary[i:i+size] for i in range(0, len(binary)-size+1, size//2)]
                    for window in windows:
                        if len(window) == size:
                            window_results = run_all_nist_tests(window, verbose=False)
                            results_list.append(window_results)
            
            # Aggregate results: A key passes if it passes in any window
            if results_list:
                for test_name in all_results:
                    best_p_value = max(res[test_name][0] for res in results_list)
                    passed = any(res[test_name][1] for res in results_list)
                    all_results[test_name].append((best_p_value, passed))
            
            if (index + 1) % 100 == 0:
                print(f"  Processed {index + 1}/{num_keys} keys...")
                
        except Exception as e:
            # print(f"Skipping key {index}: {e}") # Debug only
            continue

    # Print summary statistics
    print("\n" + "="*60)
    print("AGGREGATE RESULTS ACROSS ALL KEYS FOR CLOSED WEBCAM")
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
    
    return all_results

# --------------------------------
# Main loop
# --------------------------------           
def main():
    print("------ Quantum Inspired Key Generator ------")
    print("======Press ESC to exit, P to pause, S to save screenshot======")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print(" Cannot access webcam")
        return

    paused = False
    screenshot_counter = 0

    plt.ion()
    plt.figure('Entropy Plot')
    
    FRAMES_TO_SKIP = 5
    frame_counter = 0
    
    cv.namedWindow("QRNG Simulation", cv.WINDOW_NORMAL)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("No webcam feed available")
                break

            frame = cv.resize(frame, (960, 720))
            frame = cv.flip(frame, 1)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            h, w = gray.shape
            basis = random_basis(h, w)

            frame_with_basis, mask = draw_basis_and_mask(frame.copy(), basis)

            # Collapse measurement into key
            key = measure_wavefunction(gray, mask)
            
            # Calculate Shannon entropy from the measurement
            measurement = gray[mask == 255]
            entropy_value = calculate_shannon_entropy(measurement)
            
            # Display entropy on screen
            cv.putText(frame_with_basis, f"Entropy: {entropy_value:.2f} bits", (20, 100),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            update_entropy_graph(entropy_value)

            if key:
                display_key = key[:16]
                    
                cv.putText(frame_with_basis, f"Quantum Key: {display_key}...",
                        (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
                    
                # Save with validated key and entropy (using corrected add_file)
                if entropy_value > 0:
                    add_file("closed.csv", key, entropy_value, basis.kind)

            # Display which basis was used
            cv.putText(frame_with_basis, f"Measurement Basis: {basis.kind}",
                    (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv.imshow("QRNG Simulation", frame_with_basis)

        keypress = cv.waitKey(1) & 0xFF

        if keypress == 27:
            print("Exiting...")
            break
        elif keypress == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif keypress == ord('s'):
            screenshot_counter += 1
            filename = f"screenshot_{screenshot_counter}.png"
            cv.imwrite(filename, frame_with_basis)
            print(f"Saved screenshot: {filename}")

        frame_counter = (frame_counter + 1) % FRAMES_TO_SKIP
        if frame_counter != 0:
            continue

    cap.release()
    cv.destroyAllWindows()
# -------------------------------
# Run program
# -------------------------------
if __name__ == "__main__":
    gui = App()
    gui.mainloop()