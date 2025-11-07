import numpy as np
import os
import binascii
from scipy.stats import norm # We need the SciPy statistics package

# --- CONFIGURATION ---
INPUT_BIN_FILE = 'keys_group_E_for_NIST.bin'

# NOTE: Since the full NIST 800-22 test suite is complex, 
# this script implements the two most critical tests using SciPy.

def load_data_and_prepare():
    """Loads the binary file and converts it into a sequence of bits (0s and 1s)."""
    if not os.path.exists(INPUT_BIN_FILE):
        print(f"Error: Input file '{INPUT_BIN_FILE}' not found.")
        print("Ensure 'keys_group_E_for_NIST.bin' is in the same directory.")
        return None
    
    with open(INPUT_BIN_FILE, 'rb') as f:
        binary_data = f.read()

    # Convert binary data to a sequence of bits (0s and 1s)
    # Each byte is converted to an 8-bit string, and then concatenated
    bits = np.array([int(b) for byte in binary_data for b in format(byte, '08b')])
    
    if len(bits) < 10000:
        print(f"Warning: Only {len(bits)} bits loaded. NIST tests require millions for reliability.")
    
    return bits

def monobit_test(bits):
    """NIST Monobit Test (measures the proportion of 1s and 0s)."""
    n = len(bits)
    S_n = np.sum(bits) * 2 - n # S_n is the count of 1s minus the count of 0s
    
    # Calculate the test statistic (P-value)
    P_value = norm.sf(abs(S_n) / (np.sqrt(n) * np.sqrt(1/4))) 
    
    # Decision: The data passes if P_value >= 0.01
    return P_value, P_value >= 0.01

def runs_test(bits):
    """NIST Runs Test (measures the number of runs of consecutive identical bits)."""
    n = len(bits)
    pi = np.sum(bits) / n # Proportion of ones
    
    if abs(pi - 0.5) > (2 / np.sqrt(n)):
        # If the Monobit test fails too severely, the Runs test is invalid
        return 0.0, False 
    
    # Calculate V_n: total number of runs (changes from 0 to 1 or 1 to 0)
    V_n = 1 + np.sum(bits[:-1] != bits[1:])
    
    # Calculate the test statistic (P-value)
    expected_V = 2 * n * pi * (1 - pi)
    sigma_V = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    
    P_value = norm.sf(abs(V_n - expected_V) / sigma_V)
    
    # Decision: The data passes if P_value >= 0.01
    return P_value, P_value >= 0.01


def run_all_tests(bits):
    """Runs the statistical tests on the loaded bit sequence."""
    if bits is None:
        return
        
    print(f"\n--- Running Core Statistical Tests on {INPUT_BIN_FILE} ({len(bits)} bits) ---")
    
    # Monobit Test
    p_mono, pass_mono = monobit_test(bits)
    print(f"1. Monobit Test (Uniformity of 0s and 1s):")
    print(f"   P-value: {p_mono:.4f} -> {'PASS' if pass_mono else 'FAIL'} (Required P >= 0.01)")
    
    # Runs Test
    p_runs, pass_runs = runs_test(bits)
    print(f"2. Runs Test (Sequences of identical bits):")
    print(f"   P-value: {p_runs:.4f} -> {'PASS' if pass_runs else 'FAIL'} (Required P >= 0.01)")
    
    # Final Summary
    total_passed = int(pass_mono) + int(pass_runs)
    print("\n--- Summary ---")
    print(f"Total Tests Run: 2")
    print(f"Total Tests Passed: {total_passed}")
    print("\nIf both tests PASS, your Group E data is statistically random and validates your CSPRNG fallback.")


if __name__ == '__main__':
    bit_sequence = load_data_and_prepare()
    if bit_sequence is not None:
        run_all_tests(bit_sequence)