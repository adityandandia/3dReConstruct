import subprocess
import importlib.util
import shutil
import sys

def check_system_dependency(command_name):
    """Checks if a system command is available in the system PATH."""
    print(f"Checking for system executable: {command_name}...", end=" ")
    if shutil.which(command_name):
        print("✅ Installed")
        return True
    else:
        print("❌ Missing")
        return False

def check_python_dependency(module_name):
    """Checks if a python package is installed."""
    print(f"Checking for python package: {module_name}...", end=" ")
    if importlib.util.find_spec(module_name):
        print("✅ Installed")
        return True
    else:
        print("❌ Missing")
        return False

def main():
    print("--- System Dependencies ---")
    system_deps = [
        "ffmpeg",
        "colmap",
        "ngrok",
        "python3" # Or python, depending on your env
        # Note: FastGS might be a cloned repo rather than a global bin. 
        # Add "fastgs" here if you have it symlinked globally.
    ]
    
    sys_results = [check_system_dependency(dep) for dep in system_deps]

    print("\n--- Python Dependencies ---")
    python_deps = [
        "fastapi",
        "uvicorn",
        "open3d",
        "multipart", # <--- Fixed!
        "torch" 
    ]
    
    py_results = [check_python_dependency(dep) for dep in python_deps]

    if all(sys_results) and all(py_results):
        print("\n🚀 All dependencies are successfully installed. You are good to go!")
    else:
        print("\n⚠️ Some dependencies are missing. Please install the ones marked with ❌ before running the pipeline.")

if __name__ == "__main__":
    main()
