\
"""
run_demo.py

"""
import os
import subprocess
import sys

def main():
    # Use current python env to run streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)

if __name__ == "__main__":
    main()
