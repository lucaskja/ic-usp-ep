#!/usr/bin/env python3
"""
Debug script to check MemTorch installation and import issues.
"""

import sys
import os
import traceback

def check_python_version():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

def check_memtorch_installation():
    """Check if memtorch is installed and can be imported."""
    try:
        import memtorch
        print(f"MemTorch successfully imported from {memtorch.__file__}")
        print(f"MemTorch version: {memtorch.__version__ if hasattr(memtorch, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print(f"Error importing memtorch: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def check_memtorch_bindings():
    """Check if memtorch_bindings can be imported."""
    try:
        import memtorch_bindings
        print(f"memtorch_bindings successfully imported from {memtorch_bindings.__file__}")
        return True
    except ImportError as e:
        print(f"Error importing memtorch_bindings: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"Error importing torch: {e}")
        return False

def check_path():
    """Check Python path."""
    print("Python path:")
    for path in sys.path:
        print(f"  {path}")

def check_memtorch_cpu():
    """Check if memtorch-cpu is installed."""
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("memtorch-cpu").version
        print(f"memtorch-cpu version: {version}")
        return True
    except (pkg_resources.DistributionNotFound, ImportError) as e:
        print(f"memtorch-cpu not found: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 50)
    print("MemTorch Debug Information")
    print("=" * 50)
    
    check_python_version()
    
    print("\n" + "=" * 50)
    print("Checking Python Path")
    print("=" * 50)
    check_path()
    
    print("\n" + "=" * 50)
    print("Checking PyTorch Installation")
    print("=" * 50)
    check_pytorch()
    
    print("\n" + "=" * 50)
    print("Checking memtorch-cpu Package")
    print("=" * 50)
    check_memtorch_cpu()
    
    print("\n" + "=" * 50)
    print("Checking MemTorch Import")
    print("=" * 50)
    memtorch_available = check_memtorch_installation()
    
    if memtorch_available:
        print("\n" + "=" * 50)
        print("Checking MemTorch Bindings")
        print("=" * 50)
        check_memtorch_bindings()
    
    print("\n" + "=" * 50)
    print("Environment Variables")
    print("=" * 50)
    for key, value in os.environ.items():
        if "PATH" in key or "PYTHON" in key:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    if memtorch_available:
        print("MemTorch is available and can be imported.")
        print("If you're still seeing 'MemTorch not available' errors, check for import issues in specific modules.")
    else:
        print("MemTorch could not be imported. Check the error messages above.")
        print("Possible solutions:")
        print("1. Make sure memtorch-cpu is properly installed: pip install memtorch-cpu")
        print("2. Check if memtorch_bindings.pyd exists in your site-packages directory")
        print("3. Try importing memtorch in a simple Python script to isolate the issue")
        print("4. Check if you have the required dependencies (Visual C++ Redistributable)")

if __name__ == "__main__":
    main()
