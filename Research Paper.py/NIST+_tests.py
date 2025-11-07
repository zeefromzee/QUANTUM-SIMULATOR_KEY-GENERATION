import numpy as np
import pandas as pd
import os
import math
from scipy.special import erfc, gammaincc
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# NIST TESTS (From your original code)
# ============================================================

def nist_monobit_test(binary_string):
    """NIST Monobit (Frequency) Test"""
    n = len(binary_string)
    S_n = sum(1 if bit == '1' else -1 for bit in binary_string)
    s_obs = abs(S_n) / math.sqrt(n)
    p_value = erfc(s_obs / math.sqrt(2))
    passed = p_value >= 0.01
    return p_value, passed

def nist_runs_test(binary_string):
    """NIST Runs Test"""
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
    
    if denominator == 0: 
        return 0.0, False
    
    p_value = erfc(numerator / denominator)
    passed = p_value >= 0.01
    return p_value, passed

def nist_longest_run_test(binary_string):
    """NIST Longest Run Test"""
    n = len(binary_string)
    
    if n < 128: 
        return 0.0, False
    
    if n < 6272:
        M = 8
        K = 3
        N_blocks = 16
        v_values = [1, 2, 3, 4]
        pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
    else:
        M = 128
        K = 5
        N_blocks = n // M
        v_values = [4, 5, 6, 7, 8, 9]
        pi_values = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        N_blocks = min(N_blocks, 49)
    
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
    """NIST Spectral (DFT) Test"""
    n = len(binary_string)
    X = np.array([1 if bit == '1' else -1 for bit in binary_string])
    S = np.fft.fft(X)
    M = abs(S[:n//2])
    
    T = math.sqrt(math.log(1/0.05) * n)
    N0 = 0.95 * n / 2
    N1 = len(M[M < T])
    
    if n * 0.95 * 0.05 / 4 == 0: 
        return 0.0, False
    
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    p_value = erfc(abs(d) / math.sqrt(2))
    passed = p_value >= 0.01
    return p_value, passed

def nist_approximate_entropy_test(binary_string, m=2):
    """NIST Approximate Entropy Test"""
    n = len(binary_string)
    
    def phi(m_val):
        patterns = {}
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

# ============================================================
# ADDITIONAL TESTS
# ============================================================

def chi_square_test(binary_string):
    """Chi-Square Test for Uniformity"""
    try:
        n = len(binary_string)
        ones = binary_string.count('1')
        zeros = n - ones
        expected = n / 2
        
        chi_square = ((ones - expected)**2 / expected + 
                     (zeros - expected)**2 / expected)
        
        p_value = 1 - stats.chi2.cdf(chi_square, 1)
        passed = p_value >= 0.01
        return p_value, passed
    except:
        return 0.0, False

def serial_correlation_test(binary_string):
    """Serial Correlation Test (measures bit independence)"""
    try:
        n = len(binary_string)
        if n < 2:
            return 0.0, False
        
        bits = [int(b) for b in binary_string]
        mean = sum(bits) / n
        
        numerator = sum((bits[i] - mean) * (bits[i+1] - mean) 
                       for i in range(n-1))
        denominator = sum((bits[i] - mean)**2 for i in range(n))
        
        if denominator == 0:
            return 0.0, False
        
        correlation = numerator / denominator
        se = 1 / math.sqrt(n - 1)
        z_score = abs(correlation) / se
        
        p_value = 2 * (1 - stats.norm.cdf(z_score))
        passed = p_value >= 0.01
        return p_value, passed
    except:
        return 0.0, False

def poker_test(binary_string, m=4):
    """Poker Test (tests distribution of m-bit patterns)"""
    try:
        n = len(binary_string)
        k = n // m
        
        if k < 5:
            return 0.0, False
        
        patterns = {}
        for i in range(k):
            pattern = binary_string[i*m:(i+1)*m]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        expected = k / (2**m)
        chi_square = sum((count - expected)**2 / expected 
                        for count in patterns.values())
        
        df = 2**m - 1
        p_value = 1 - stats.chi2.cdf(chi_square, df)
        passed = p_value >= 0.01
        return p_value, passed
    except:
        return 0.0, False

def gap_test(binary_string, target='1'):
    """Gap Test (tests distribution of gaps between occurrences)"""
    try:
        gaps = []
        current_gap = 0
        in_gap = False
        
        for bit in binary_string:
            if bit == target:
                if in_gap:
                    gaps.append(current_gap)
                    current_gap = 0
                in_gap = True
            else:
                if in_gap:
                    current_gap += 1
        
        if len(gaps) < 5:
            return 0.0, False
        
        categories = [0, 1, 2, 3, 4, 5]
        observed = [0] * len(categories)
        
        for gap in gaps:
            if gap >= len(categories) - 1:
                observed[-1] += 1
            else:
                observed[gap] += 1
        
        total = len(gaps)
        expected = []
        for i in range(len(categories) - 1):
            expected.append(total * (0.5 ** (i + 1)))
        expected.append(total - sum(expected))
        
        chi_square = sum((obs - exp)**2 / exp 
                        for obs, exp in zip(observed, expected) if exp > 0)
        
        df = len(categories) - 1
        p_value = 1 - stats.chi2.cdf(chi_square, df)
        passed = p_value >= 0.01
        return p_value, passed
    except:
        return 0.0, False

def autocorrelation_test(binary_string, lag=1):
    """Autocorrelation Test (tests correlation at different lags)"""
    try:
        n = len(binary_string)
        if n < lag + 10:
            return 0.0, False
        
        bits = np.array([int(b) for b in binary_string])
        mean = bits.mean()
        c0 = np.sum((bits - mean) ** 2)
        
        if c0 == 0:
            return 0.0, False
        
        c_lag = np.sum((bits[:-lag] - mean) * (bits[lag:] - mean))
        autocorr = c_lag / c0
        
        se = 1 / math.sqrt(n)
        z_score = abs(autocorr) / se
        
        p_value = 2 * (1 - stats.norm.cdf(z_score))
        passed = p_value >= 0.01
        return p_value, passed
    except:
        return 0.0, False

def entropy_test(binary_string):
    """Shannon Entropy Test"""
    try:
        n = len(binary_string)
        if n == 0:
            return 0.0, False
        
        ones = binary_string.count('1')
        zeros = n - ones
        
        if ones == 0 or zeros == 0:
            return 0.0, False
        
        p1 = ones / n
        p0 = zeros / n
        
        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        p_value = entropy
        passed = entropy >= 0.95
        
        return p_value, passed
    except:
        return 0.0, False

def block_frequency_test(binary_string, block_size=128):
    """Block Frequency Test (NIST)"""
    try:
        n = len(binary_string)
        N = n // block_size
        
        if N < 1:
            return 0.0, False
        
        proportions = []
        for i in range(N):
            block = binary_string[i*block_size:(i+1)*block_size]
            pi = block.count('1') / block_size
            proportions.append(pi)
        
        chi_square = 4 * block_size * sum((pi - 0.5)**2 for pi in proportions)
        
        p_value = gammaincc(N/2, chi_square/2)
        passed = p_value >= 0.01
        return p_value, passed
    except:
        return 0.0, False

def cumulative_sums_test(binary_string, mode='forward'):
    """Cumulative Sums Test (NIST)"""
    try:
        n = len(binary_string)
        X = [1 if bit == '1' else -1 for bit in binary_string]
        
        if mode == 'forward':
            cumsum = [sum(X[:i+1]) for i in range(n)]
        else:
            cumsum = [sum(X[i:]) for i in range(n)]
        
        z = max(abs(s) for s in cumsum)
        
        sum_term = 0
        for k in range(int((-n/z + 1)/4), int((n/z - 1)/4) + 1):
            sum_term += (stats.norm.cdf((4*k + 1)*z/math.sqrt(n)) - 
                        stats.norm.cdf((4*k - 1)*z/math.sqrt(n)))
        
        for k in range(int((-n/z - 3)/4), int((n/z - 1)/4) + 1):
            sum_term -= (stats.norm.cdf((4*k + 3)*z/math.sqrt(n)) - 
                        stats.norm.cdf((4*k + 1)*z/math.sqrt(n)))
        
        p_value = 1 - sum_term
        passed = p_value >= 0.01
        return p_value, passed
    except:
        return 0.0, False

# ============================================================
# RUN ALL TESTS (Single window)
# ============================================================

def run_all_tests(binary_string):
    """Run all randomness tests on a binary string"""
    tests = {
        # NIST Tests
        "NIST Monobit": nist_monobit_test,
        "NIST Runs": nist_runs_test,
        "NIST Longest Run": nist_longest_run_test,
        "NIST Spectral": nist_spectral_test,
        "NIST Approximate Entropy": nist_approximate_entropy_test,
        "NIST Block Frequency": block_frequency_test,
        "NIST Cumulative Sums": cumulative_sums_test,
        
        # Additional Statistical Tests
        "Chi-Square": chi_square_test,
        "Serial Correlation": serial_correlation_test,
        "Poker Test": poker_test,
        "Gap Test": gap_test,
        "Autocorrelation": autocorrelation_test,
        "Shannon Entropy": entropy_test,
    }
    
    results = {}
    for test_name, test_func in tests.items():
        try:
            p_value, passed = test_func(binary_string)
            results[test_name] = {
                'p_value': p_value,
                'passed': passed,
                'status': 'PASS' if passed else 'FAIL'
            }
        except Exception as e:
            results[test_name] = {
                'p_value': 0.0,
                'passed': False,
                'status': 'ERROR'
            }
    
    return results

# ============================================================
# CSV PROCESSING
# ============================================================

def load_keys_from_csv(csv_file):
    """Load and extract keys from CSV file"""
    try:
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found.")
            return None
        
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("Error: CSV file is empty.")
            return None
        
        if 'Key Generated' in df.columns:
            df['clean_key'] = df['Key Generated'].astype(str).str.extract(r'Quantum Key: ([0-9a-fA-F]+)')
        else:
            print("Error: 'Key Generated' column not found.")
            return None
        
        df = df[df['clean_key'].str.len() >= 2].copy()
        
        if df.empty:
            print("No valid keys found.")
            return None
        
        print(f"Loaded {len(df)} valid keys from {csv_file}")
        return df['clean_key'].tolist()
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def hex_to_binary(hex_string):
    """Convert hex string to binary string"""
    try:
        return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)
    except:
        return None

def xor_fold_for_test(key_hex):
    """XOR fold for better test distribution (OPTIONAL preprocessing)"""
    try:
        data = bytes.fromhex(key_hex)
        mid = len(data) // 2
        first_half = data[:mid]
        second_half = data[mid:mid + len(first_half)]
        xored = bytes(a ^ b for a, b in zip(first_half, second_half))
        return xored.hex()
    except:
        return key_hex

# ============================================================
# MAIN TESTING FUNCTION (MATCHING YOUR ORIGINAL METHODOLOGY)
# ============================================================

def comprehensive_randomness_test(csv_file, max_keys=None, detailed_output=False, 
                                  use_windowing=True, use_xor_fold=True):
    """
    Run comprehensive randomness tests on keys from CSV file
    
    Parameters:
    - csv_file: Path to CSV file ('enc.csv' or 'closed.csv')
    - max_keys: Maximum number of keys to test (None = all)
    - detailed_output: If True, show results for each key
    - use_windowing: If True, use multiple window testing (like original code)
    - use_xor_fold: If True, apply XOR folding before testing
    """
    
    print("\n" + "="*70)
    print(f"COMPREHENSIVE RANDOMNESS TEST SUITE")
    print(f"Testing file: {csv_file}")
    print(f"Windowing: {use_windowing} | XOR Fold: {use_xor_fold}")
    print("="*70)
    
    keys = load_keys_from_csv(csv_file)
    if keys is None:
        return
    
    if max_keys is not None:
        keys = keys[:max_keys]
    
    num_keys = len(keys)
    print(f"\nTesting {num_keys} keys...\n")
    
    # Aggregate results
    all_results = {}
    valid_tests = 0
    
    # Test each key
    for idx, key_hex in enumerate(keys):
        try:
            # Optional XOR folding (matches your original test)
            if use_xor_fold:
                key_hex = xor_fold_for_test(key_hex)
            
            binary = hex_to_binary(key_hex)
            if binary is None or len(binary) < 100:
                continue
            
            # WINDOWED TESTING (matching your original code)
            if use_windowing:
                results_list = []
                window_sizes = [256, 128, 64]
                
                for size in window_sizes:
                    if size <= len(binary):
                        # Test different windows with 50% overlap
                        windows = [binary[i:i+size] for i in range(0, len(binary)-size+1, size//2)]
                        for window in windows:
                            if len(window) == size:
                                window_results = run_all_tests(window)
                                results_list.append(window_results)
                
                # Aggregate: key passes if it passes in ANY window
                if results_list:
                    key_results = {}
                    for test_name in results_list[0].keys():
                        best_p_value = max(res[test_name]['p_value'] for res in results_list)
                        passed = any(res[test_name]['passed'] for res in results_list)
                        key_results[test_name] = {
                            'p_value': best_p_value,
                            'passed': passed,
                            'status': 'PASS' if passed else 'FAIL'
                        }
                else:
                    continue
            else:
                # Single test on entire binary string
                key_results = run_all_tests(binary)
            
            if detailed_output:
                print(f"\n--- Key {idx + 1} ---")
                for test_name, result in key_results.items():
                    status = "✓" if result['passed'] else "✗"
                    print(f"{status} {test_name}: p={result['p_value']:.4f}")
            
            # Aggregate results
            for test_name, result in key_results.items():
                if test_name not in all_results:
                    all_results[test_name] = {
                        'p_values': [],
                        'passes': 0,
                        'fails': 0,
                        'errors': 0
                    }
                
                all_results[test_name]['p_values'].append(result['p_value'])
                if result['status'] == 'PASS':
                    all_results[test_name]['passes'] += 1
                elif result['status'] == 'FAIL':
                    all_results[test_name]['fails'] += 1
                else:
                    all_results[test_name]['errors'] += 1
            
            valid_tests += 1
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{num_keys} keys...")
                
        except Exception as e:
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("AGGREGATE TEST RESULTS")
    print("="*70)
    
    summary_data = []
    
    for test_name, results in all_results.items():
        if results['p_values']:
            avg_p = np.mean(results['p_values'])
            min_p = np.min(results['p_values'])
            max_p = np.max(results['p_values'])
            std_p = np.std(results['p_values'])
            pass_rate = (results['passes'] / valid_tests) * 100
            
            summary_data.append({
                'Test': test_name,
                'Avg p-value': avg_p,
                'Min p-value': min_p,
                'Max p-value': max_p,
                'Std Dev': std_p,
                'Pass Rate': pass_rate,
                'Passes': results['passes'],
                'Fails': results['fails']
            })
            
            print(f"\n{test_name}:")
            print(f"  Average p-value:  {avg_p:.4f}")
            print(f"  Min p-value:      {min_p:.4f}")
            print(f"  Max p-value:      {max_p:.4f}")
            print(f"  Std Dev:          {std_p:.4f}")
            print(f"  Pass rate:        {pass_rate:.1f}% ({results['passes']}/{valid_tests})")
    
    # Overall statistics
    total_tests = len(all_results)
    avg_pass_rate = np.mean([data['Pass Rate'] for data in summary_data])
    
    print("\n" + "="*70)
    print(f"OVERALL SUMMARY")
    print("="*70)
    print(f"Total keys tested:        {valid_tests}")
    print(f"Total tests performed:    {total_tests}")
    print(f"Average pass rate:        {avg_pass_rate:.1f}%")
    print("="*70 + "\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save results
    output_file = csv_file.replace('.csv', '_test_results.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}\n")
    
    return summary_df, all_results

# ============================================================
# INTERACTIVE MENU
# ============================================================

def main_menu():
    """Interactive menu for choosing test options"""
    print("\n" + "="*70)
    print("QUANTUM KEY RANDOMNESS TEST SUITE")
    print("="*70)
    print("\nSelect CSV file to test:")
    print("1. enc.csv (Open Camera - Environment Noise)")
    print("2. closed.csv (Closed Camera - Dark Noise)")
    print("3. Custom file path")
    print("4. Compare both files")
    print("5. Test with different methodologies (windowed vs full)")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    if choice == '0':
        print("Exiting...")
        return
    
    elif choice in ['1', '2', '3']:
        if choice == '1':
            csv_file = 'enc.csv'
        elif choice == '2':
            csv_file = 'closed.csv'
        else:
            csv_file = input("Enter CSV file path: ").strip()
        
        max_keys = input("Max keys to test (Enter for all): ").strip()
        max_keys = int(max_keys) if max_keys else None
        
        detailed = input("Show detailed output? (y/n): ").strip().lower() == 'y'
        
        windowing = input("Use windowing (like original code)? (y/n): ").strip().lower() != 'n'
        xor_fold = input("Apply XOR folding? (y/n): ").strip().lower() != 'n'
        
        comprehensive_randomness_test(csv_file, max_keys, detailed, windowing, xor_fold)
    
    elif choice == '4':
        print("\n--- Testing enc.csv (Open Camera) ---")
        comprehensive_randomness_test('enc.csv', max_keys=100, use_windowing=True, use_xor_fold=True)
        
        print("\n--- Testing closed.csv (Closed Camera) ---")
        comprehensive_randomness_test('closed.csv', max_keys=100, use_windowing=True, use_xor_fold=True)
    
    elif choice == '5':
        csv_file = input("Enter CSV file (enc.csv/closed.csv): ").strip()
        
        print("\n--- METHOD 1: With Windowing + XOR Fold (Original) ---")
        comprehensive_randomness_test(csv_file, max_keys=100, use_windowing=True, use_xor_fold=True)
        
        print("\n--- METHOD 2: No Windowing, No XOR Fold (Raw) ---")
        comprehensive_randomness_test(csv_file, max_keys=100, use_windowing=False, use_xor_fold=False)
    
    else:
        print("Invalid choice. Please try again.")
        main_menu()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main_menu()
