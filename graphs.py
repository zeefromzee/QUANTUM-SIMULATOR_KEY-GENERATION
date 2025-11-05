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
import math
from scipy.special import erfc, gammaincc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class App(CTk.CTk):
    def __init__(self):
        super().__init__()  
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
        self.button_graphs()
        self.button_migrate()
    
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

    def button_graphs(self):
        button = CTk.CTkButton(self.main_frame, 
                        text="Generate Advanced Graphs", 
                        command=lambda: generate_advanced_graphs("enc.csv"))
        button.pack(pady=40)

    def button_migrate(self):
        """Button to migrate old CSV files."""
        button = CTk.CTkButton(
            self.main_frame, 
            text="Migrate Old CSV File", 
            command=lambda: migrate_old_csv("enc.csv")
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

def load_entropy_data(csv_file="enc.csv"):
    """
    Load entropy data from CSV with comprehensive error handling.
    Validates file structure and extracts numeric values.
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file '{csv_file}' not found. Generate keys first.")
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Check if empty
        if df.empty:
            raise ValueError(f"CSV file '{csv_file}' is empty. No data to process.")
        
        # Verify required columns exist
        required_cols = ['ID', 'Key Generated', 'Entropy Value', 'Basis Kind']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            available = list(df.columns)
            raise KeyError(
                f"Missing columns: {missing_cols}\n"
                f"Available columns: {available}\n"
                f"Hint: You may need to regenerate keys to get the updated CSV format."
            )
        
        # Extract numeric entropy value from "Entropy: X.XX bits" format
        try:
            df['Entropy Value'] = df['Entropy Value'].str.extract(r'([\d\.]+)').astype(float)
        except Exception as e:
            raise ValueError(f"Failed to parse Entropy Value column: {e}")
        
        # Check for any NaN values in critical columns
        if df['Entropy Value'].isna().any():
            nan_count = df['Entropy Value'].isna().sum()
            print(f"[Warning] Found {nan_count} invalid entropy values. These will be skipped.")
            df = df.dropna(subset=['Entropy Value'])
        
        print(f"[✓] Loaded {len(df)} valid records from {csv_file}")
        return df
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return None
    except KeyError as e:
        print(f"[Error] {e}")
        return None
    except ValueError as e:
        print(f"[Error] {e}")
        return None
    except Exception as e:
        print(f"[Unexpected Error] Failed to load CSV: {e}")
        return None


def validate_csv_structure(csv_file="enc.csv"):
    """
    Validates CSV file structure before processing.
    Returns True if valid, False otherwise.
    """
    try:
        if not os.path.exists(csv_file):
            print(f"[✗] File not found: {csv_file}")
            return False
        
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print(f"[✗] File is empty: {csv_file}")
            return False
        
        required_cols = ['ID', 'Key Generated', 'Entropy Value', 'Basis Kind']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"[✗] Missing required columns: {missing_cols}")
            print(f"[Info] Current columns: {list(df.columns)}")
            print(f"[Hint] Delete {csv_file} and regenerate keys to get the updated format.")
            return False
        
        print(f"[✓] CSV structure is valid ({len(df)} records)")
        return True
        
    except Exception as e:
        print(f"[✗] Validation failed: {e}")
        return False


def plot_entropy_over_time(df, save_path="entropy_over_time.png"):
    """Entropy over time line graph with error handling."""
    try:
        if df is None or df.empty:
            print("[!] No data available for Entropy Over Time plot.")
            return False
        
        plt.figure(figsize=(8, 4))
        plt.plot(df['ID'], df['Entropy Value'], marker='o', linestyle='-', alpha=0.7)
        plt.title("Entropy Over Time")
        plt.xlabel("Frame / Key ID")
        plt.ylabel("Entropy (bits)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"[✓] Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to generate Entropy Over Time plot: {e}")
        return False


def plot_entropy_histogram(df, save_path="entropy_histogram.png"):
    """Entropy histogram with error handling."""
    try:
        if df is None or df.empty:
            print("[!] No data available for Entropy Histogram.")
            return False
        
        plt.figure(figsize=(6, 4))
        plt.hist(df['Entropy Value'], bins=15, color='skyblue', edgecolor='black')
        plt.title("Distribution of Entropy Values")
        plt.xlabel("Entropy (bits)")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"[✓] Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to generate Entropy Histogram: {e}")
        return False


def plot_bit_proportion_histogram(df, save_path="bit_proportion_histogram.png"):
    """Bit proportion histogram with enhanced error handling."""
    try:
        if df is None or df.empty:
            print("[!] No data available for Bit Proportion plot.")
            return False
        
        def hex_to_bit_proportion(key_full_string):
            try:
                # Extract hex from "Quantum Key: <hex_string>"
                key_hex = key_full_string.split(':')[1].strip().replace('"', '')
                # Convert to binary
                binary_string = bin(int(key_hex, 16))[2:].zfill(len(key_hex) * 4)
                # Calculate proportion
                count_ones = binary_string.count('1')
                total_bits = len(binary_string)
                return count_ones / total_bits if total_bits > 0 else None
            except Exception as e:
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
        print(f"[✓] Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to generate Bit Proportion plot: {e}")
        return False


def plot_entropy_by_basis_boxplot(df, save_path="entropy_by_basis.png"):
    """Box plot of entropy by basis with error handling."""
    try:
        if df is None or df.empty:
            print("[!] No data available for Entropy by Basis plot.")
            return False
        
        if 'Basis Kind' not in df.columns:
            print("[!] Cannot plot Entropy by Basis: 'Basis Kind' column missing.")
            print("[Hint] Delete enc.csv and regenerate keys to get the updated format.")
            return False
        
        # Check if we have data for multiple basis types
        unique_bases = df['Basis Kind'].unique()
        if len(unique_bases) < 2:
            print(f"[Warning] Only {len(unique_bases)} basis type(s) found. Box plot works best with multiple types.")
        
        plt.figure(figsize=(6, 4))
        df.boxplot(column='Entropy Value', by='Basis Kind', grid=True, patch_artist=True)
        plt.suptitle('')
        plt.title("Entropy Distribution by Measurement Basis")
        plt.xlabel("Measurement Basis Type")
        plt.ylabel("Entropy (bits)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"[✓] Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to generate Entropy by Basis plot: {e}")
        return False


def generate_advanced_graphs(csv_file="enc.csv"):
    """
    Master function to generate all advanced graphs with comprehensive error handling.
    """
    print("\n" + "="*60)
    print("ADVANCED GRAPH GENERATION")
    print("="*60)
    
    # Step 1: Validate CSV structure
    print("\n[1/4] Validating CSV structure...")
    if not validate_csv_structure(csv_file):
        print("\n[✗] Graph generation aborted due to validation errors.")
        print("[Action Required] Please regenerate keys to create a valid CSV file.")
        return False
    
    # Step 2: Load data
    print("\n[2/4] Loading data...")
    df = load_entropy_data(csv_file)
    if df is None:
        print("\n[✗] Failed to load data. Graph generation aborted.")
        return False
    
    # Step 3: Generate graphs
    print("\n[3/4] Generating graphs...")
    results = {
        "Bit Proportion Histogram": plot_bit_proportion_histogram(df.copy()),
        "Entropy by Basis": plot_entropy_by_basis_boxplot(df.copy()),
        "Entropy Over Time": plot_entropy_over_time(df.copy()),
        "Entropy Histogram": plot_entropy_histogram(df.copy())
    }
    
    # Step 4: Summary
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

# ============================================================
# CSV MIGRATION SCRIPT - Add missing Basis Kind column
# ============================================================

def migrate_old_csv(old_file="enc.csv", backup=True):
    """
    Migrates old CSV format to new format by adding 'Basis Kind' column.
    Randomly assigns basis types to existing records since we can't recover the original.
    """
    try:
        import pandas as pd
        import secrets
        import shutil
        from datetime import datetime
        
        print("\n" + "="*60)
        print("CSV MIGRATION TOOL")
        print("="*60)
        
        # Check if file exists
        if not os.path.exists(old_file):
            print(f"[✗] File not found: {old_file}")
            return False
        
        # Backup original file
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{old_file}.backup_{timestamp}"
            shutil.copy(old_file, backup_file)
            print(f"[✓] Created backup: {backup_file}")
        
        # Load old CSV
        print(f"[→] Reading {old_file}...")
        df = pd.read_csv(old_file)
        
        # Check if Basis Kind already exists
        if 'Basis Kind' in df.columns:
            print("[!] 'Basis Kind' column already exists. No migration needed.")
            return True
        
        # Add Basis Kind column with random values
        # (We can't know the original basis, so we assign randomly)
        print("[→] Adding 'Basis Kind' column...")
        basis_types = ['rect', 'circle', 'ellipse']
        df['Basis Kind'] = [secrets.choice(basis_types) for _ in range(len(df))]
        
        # Save updated CSV
        print(f"[→] Writing updated file...")
        df.to_csv(old_file, index=False)
        
        print(f"[✓] Migration complete! Updated {len(df)} records.")
        print(f"[Note] Basis Kind values are randomly assigned since original data is unavailable.")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"[✗] Migration failed: {e}")
        return False


# ============================================================
# ALTERNATIVE: Generate sample data for testing
# ============================================================

def generate_sample_csv(filename="enc.csv", num_samples=50):
    """
    Generates sample CSV data for testing graphs.
    Useful if you want to test without running the full key generation.
    """
    try:
        import secrets
        import hashlib
        
        print(f"\n[→] Generating {num_samples} sample records...")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Key Generated", "Entropy Value", "Basis Kind"])
            
            basis_types = ['rect', 'circle', 'ellipse']
            
            for i in range(1, num_samples + 1):
                # Generate random key
                random_bytes = secrets.token_bytes(32)
                key = hashlib.sha256(random_bytes).hexdigest()
                
                # Random entropy between 6.5 and 8.0 bits (realistic for 8-bit pixels)
                entropy = 6.5 + (secrets.randbelow(150) / 100)
                
                # Random basis
                basis = secrets.choice(basis_types)
                
                writer.writerow([
                    i,
                    f"Quantum Key: {key}",
                    f"Entropy: {entropy:.2f} bits",
                    basis
                ])
        
        print(f"[✓] Generated {filename} with {num_samples} sample records")
        return True
        
    except Exception as e:
        print(f"[✗] Failed to generate sample data: {e}")
        return False


# ============================================================
# ENHANCED add_file WITH ERROR HANDLING
# ============================================================

count = 0

def add_file(file, key, entropy_value, basis_kind, reset=False):
    """
    Append quantum key data to CSV with error handling.
    """
    global count

    try:
        # Determine write mode
        mode = "w" if reset or not os.path.exists(file) else "a"
        
        with open(file, mode, newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if mode == "w":
                writer.writerow(["ID", "Key Generated", "Entropy Value", "Basis Kind"])
                count = 0  # Reset counter for new file

            count += 1
            writer.writerow([
                count,
                f"Quantum Key: {key}",  
                f"Entropy: {entropy_value:.2f} bits",
                basis_kind
            ])
            
    except PermissionError:
        print(f"[Error] Permission denied: Cannot write to {file}. Close it if open.")
    except IOError as e:
        print(f"[Error] File I/O error: {e}")
    except Exception as e:
        print(f"[Error] Failed to write to CSV: {e}")

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
# def main():
#     print("------ Quantum Inspired Key Generator ------")
#     print("This application is still under development so~Please look forward to newer updates :D")
#     print("======Press ESC to exit, P to pause, S to save screenshot======")


#     # Try to open webcam
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         print(" Cannot access webcam")
#         return

#     paused = False                # start in running mode
#     screenshot_counter = 0        # keep track of screenshots saved

#     plt.ion()
    
#     # Frame skipping control
#     FRAMES_TO_SKIP = 5  # Adjust this value as needed
#     frame_counter = 0

#     # Throttling control
#     M = 10  # Adjust this value for throttling frequency
#     processed_frames = 0
    
#     while True:
#         if not paused:
#             # Grab a frame from webcam
#             ret, frame = cap.read()
#             if not ret:
#                 print("No webcam feed available")
#                 break

#             # Resize feed → make video bigger
#             frame = cv.resize(frame, (960, 720))

#             # Flip horizontally → mirror-like display
#             frame = cv.flip(frame, 1)

#             # Convert to grayscale → pixel brightness values
#             gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#             # Pick a random measurement basis (shape)
#             h, w = gray.shape
#             basis = random_basis(h, w)

#             # Draw shape and get mask
#             frame_with_basis, mask = draw_basis_and_mask(frame.copy(), basis)

#             # Collapse measurement into key
#             key = measure_wavefunction(gray, mask)

        
            
#             measurement = gray[mask == 255]
#             entropy_value = calculate_shannon_entropy(measurement)
#             cv.putText(frame_with_basis, f"Entropy: {entropy_value:.2f} bits", (20, 100),
#                         cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
#             update_entropy_graph(entropy_value)

#             # If a key was generated, show part of it on screen
#             if key:
#                 cv.putText(frame_with_basis, f"Quantum Key: {key[:16]}",
#                         (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

#                 entropy_val = len(key) * 4  # just a toy measure (SHA-256 = 256 bits)
#                 add_file("enc.csv", key, entropy_val, basis.kind) # <<< CORRECTED CALL IS HERE

#             # Display which basis was used
#             cv.putText(frame_with_basis, f"Measurement Basis: {basis.kind}",
#                     (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#             # Show final output
#             cv.imshow("QRNG Simulation", frame_with_basis)

#         # -------------------------------
#         # Key controls
#         # -------------------------------
#         keypress = cv.waitKey(33) & 0xFF  # 33ms for ~30 FPS, realistic inter-frame delay

#         # Optional: Implement frame skipping to reduce load
#         frame_counter = (frame_counter + 1) % FRAMES_TO_SKIP
#         if frame_counter != 0:
#             continue

#         # Optional: Throttle CSV writes and matplotlib updates
#         if processed_frames % M == 0:
#             # Perform batched writes or updates here
#             pass

#         if keypress == 27:  # ESC = quit
#             break
#         elif keypress == ord('p'):  # P = pause/resume
#             paused = not paused
#             print("Paused" if paused else "Resumed")
#         elif keypress == ord('s') and not paused:  # S = save screenshot
#             filename = f"screenshot_{int(time.time())}.png"
#             cv.imwrite(filename, frame_with_basis)
#             print(f"Screenshot saved: {filename}")

#     # Cleanup after loop
#     cap.release()
#     cv.destroyAllWindows()
def main():
    print("------ Quantum Inspired Key Generator ------")
    print("This application is still under development so~Please look forward to newer updates :D")
    print("======Press ESC to exit, P to pause, S to save screenshot======")

    # Try to open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print(" Cannot access webcam")
        return

    paused = False
    screenshot_counter = 0

    plt.ion()
    
    # Frame skipping control
    FRAMES_TO_SKIP = 5
    frame_counter = 0

    # Throttling control
    M = 10
    processed_frames = 0
    
    while True:
        if not paused:
            # Grab a frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("No webcam feed available")
                break

            # Resize and flip
            frame = cv.resize(frame, (960, 720))
            frame = cv.flip(frame, 1)

            # Convert to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Pick a random measurement basis
            h, w = gray.shape
            basis = random_basis(h, w)

            # Draw shape and get mask
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

            # If a key was generated, save it with the CORRECT entropy value
            if key:
                cv.putText(frame_with_basis, f"Quantum Key: {key[:16]}",
                        (20, 40), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

                # ✅ FIXED: Use the actual Shannon entropy, not key length!
                add_file("enc.csv", key, entropy_value, basis.kind)

            # Display which basis was used
            cv.putText(frame_with_basis, f"Measurement Basis: {basis.kind}",
                    (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show final output
            cv.imshow("QRNG Simulation", frame_with_basis)

        # Key controls
        keypress = cv.waitKey(33) & 0xFF

        # Frame skipping
        frame_counter = (frame_counter + 1) % FRAMES_TO_SKIP
        if frame_counter != 0:
            continue

        # Throttle operations
        if processed_frames % M == 0:
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

# -------------------------------
# Run program
# -------------------------------

if __name__ == "__main__":
    gui = App()
    gui.mainloop()
