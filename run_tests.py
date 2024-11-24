import os
import subprocess

# This is needed to run each test in separate processes as the keras backend cannot be changed while set in a process

def run_test(file):
    result = subprocess.run(['pytest', f'tests/test_{backend}.py'], capture_output=True, text=True)
    print(f"\n--- {backend.upper()} Backend Test Results ---")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    return result.returncode

if __name__ == "__main__":
    backends = ['tensorflow', 'torch', 'jax']
    exit_codes = []

    for backend in backends:
        exit_codes.append(run_test(backend))

    if any(exit_codes):
        exit(1)
    else:
        print("\nAll tests passed successfully!")
        exit(0)