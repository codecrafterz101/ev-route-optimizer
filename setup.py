#!/usr/bin/env python3
"""
Setup script for EV Route Optimization System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

VENV_NAME = "ev_route_env"
PYTHON_CANDIDATES = ["python3.11", "python3.10"]


def find_python():
    """Find a suitable Python executable (3.10 or 3.11)"""
    for py in PYTHON_CANDIDATES:
        try:
            out = subprocess.check_output([py, "--version"], stderr=subprocess.STDOUT)
            version = out.decode().strip().split()[1]
            major, minor, *_ = version.split(".")
            if int(major) == 3 and int(minor) in (10, 11):
                print(f"   ✅ Found {py} ({version})")
                return py
        except Exception:
            continue
    print("❌ Python 3.10 or 3.11 is required. Please install it and try again.")
    sys.exit(1)


def main():
    print("🚗 EV Route Optimization System Setup")
    print("=" * 50)

    # Find suitable Python version
    python_exec = find_python()

    # Create virtual environment
    create_venv(python_exec)

    # Install dependencies in venv
    pip_install_requirements(python_exec)

    # Setup environment
    setup_environment()

    # Create necessary directories
    create_directories()

    # Test installation in venv
    test_installation(python_exec)

    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"   1. Activate the venv: source {VENV_NAME}/bin/activate")
    print("   2. Copy .env.example to .env and add your API keys")
    print("   3. Run: python demo_data_integration.py")
    print("   4. Check the demo_exports/ directory for results")


def create_venv(python_exec):
    print(f"\n🐍 Creating virtual environment ({VENV_NAME})...")
    if not os.path.exists(VENV_NAME):
        subprocess.check_call([python_exec, "-m", "venv", VENV_NAME])
        print(f"   ✅ Virtual environment created: {VENV_NAME}/")
    else:
        print(f"   ✅ Virtual environment already exists: {VENV_NAME}/")


def pip_install_requirements(python_exec):
    print("\n📦 Installing dependencies in venv...")
    pip_path = os.path.join(VENV_NAME, "bin", "pip")
    try:
        subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
        print("   ✅ Dependencies installed successfully in venv")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        sys.exit(1)


def setup_environment():
    print("\n⚙️  Setting up environment...")
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("   ✅ Created .env file from .env.example")
            print("   ⚠️  Please edit .env file and add your API keys")
        else:
            print("   ⚠️  No .env.example found, creating basic .env file")
            create_basic_env()
    else:
        print("   ✅ .env file already exists")


def create_basic_env():
    env_content = """# Database Configuration
DATABASE_URL=postgresql://localhost/ev_routing
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ev_routing
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# API Keys (Add your actual API keys here)
OPENWEATHER_API_KEY=your_openweather_api_key_here
TOMTOM_API_KEY=your_tomtom_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
"""
    with open('.env', 'w') as f:
        f.write(env_content)


def create_directories():
    print("\n📁 Creating directories...")
    directories = [
        'data_exports',
        'demo_exports',
        'logs',
        'models',
        'data'
    ]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")


def test_installation(python_exec):
    print("\n🧪 Testing installation in venv...")
    python_path = os.path.join(VENV_NAME, "bin", "python")
    try:
        subprocess.check_call([
            python_path, "-c",
            "import pandas, geopandas, numpy, folium, requests; print('   ✅ All required packages imported successfully')"
        ])
        subprocess.check_call([
            python_path, "-c",
            "from config import Config; from data_integration.data_manager import DataManager; config = Config(); data_manager = DataManager(config); print('   ✅ Data manager initialized successfully')"
        ])
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Setup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 