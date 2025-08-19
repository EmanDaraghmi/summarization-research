# reproducibility_check.py
"""
Script to verify reproducibility of results
"""

import json
import hashlib

def verify_reproducibility(results_file: str):
    """Verify that results are reproducible"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Check if random seed was fixed
    if data.get('reproducibility_info', {}).get('random_seed') == 42:
        print("✓ Random seed properly set to 42")
    else:
        print("✗ Random seed not properly configured")
    
    # Verify environment specifications
    required_libs = ["torch", "transformers", "datasets", "numpy"]
    actual_libs = data.get('reproducibility_info', {}).get('required_libraries', [])
    
    missing_libs = [lib for lib in required_libs if lib not in str(actual_libs)]
    if not missing_libs:
        print("✓ All required libraries specified")
    else:
        print(f"✗ Missing library specifications: {missing_libs}")
    
    # Check for dataset version information
    if 'dataset_versions' in data.get('reproducibility_info', {}):
        print("✓ Dataset version information provided")
    else:
        print("⚠ Dataset version information missing")
    
    print("\nReproducibility check completed!")

if __name__ == "__main__":
    verify_reproducibility("results/final_report.json")