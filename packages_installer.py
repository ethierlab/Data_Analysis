import subprocess
import sys

def install(package):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
packages = ["numpy", "scipy", "pandas", "ipywidgets", "IPython", "matplotlib", "jupyter"]

if __name__ == '__main__':
    install(packages)