#!/usr/bin/env python3
"""
Startup script for EV Route Optimizer Web Application
Handles environment setup and starts the Flask server
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🚗⚡ EV Route Optimizer - Web Application ⚡🚗           ║
    ║                                                              ║
    ║     AI-Powered Energy-Efficient Navigation System           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'geopandas', 'osmnx', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'static', 'templates', 'logs', 'models', 
        'results', 'cache', 'demo_exports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Required directories created/verified")

def check_env_file():
    """Check if .env file exists and create a sample if not"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  .env file not found, creating sample...")
        
        sample_env = """# EV Route Optimizer Configuration

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production

# API Keys (optional for demo)
WEATHER_API_KEY=your_weather_api_key_here
TRAFFIC_API_KEY=your_traffic_api_key_here

# Database Configuration (optional)
DATABASE_URL=sqlite:///ev_optimizer.db

# Application Settings
LOG_LEVEL=INFO
MAX_ROUTE_SEGMENTS=100
CACHE_TIMEOUT=3600
"""
        
        with open('.env', 'w') as f:
            f.write(sample_env)
        
        print("✅ Sample .env file created")
    else:
        print("✅ .env file found")

def start_flask_app():
    """Start the Flask application"""
    print("\n🚀 Starting EV Route Optimizer Web Application...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("\n" + "="*60)
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("   Make sure app.py exists and is properly configured")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down EV Route Optimizer...")
        print("   Thank you for using our application!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

def show_help():
    """Show help information"""
    help_text = """
Usage: python start_webapp.py [options]

Options:
  --help, -h     Show this help message
  --check        Check system requirements only
  --production   Start in production mode (requires gunicorn)

Examples:
  python start_webapp.py           # Start in development mode
  python start_webapp.py --check   # Check requirements only
  python start_webapp.py --production  # Start in production mode

For more information, see WEB_APP_README.md
"""
    print(help_text)

def start_production():
    """Start in production mode using gunicorn"""
    try:
        import gunicorn
    except ImportError:
        print("❌ Gunicorn not installed. Install with: pip install gunicorn")
        sys.exit(1)
    
    print("🚀 Starting in production mode...")
    cmd = ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
    subprocess.run(cmd)

def main():
    """Main function"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            show_help()
            return
        elif sys.argv[1] == '--check':
            print_banner()
            check_python_version()
            check_dependencies()
            create_directories()
            check_env_file()
            print("\n✅ System check completed successfully!")
            return
        elif sys.argv[1] == '--production':
            print_banner()
            check_python_version()
            if not check_dependencies():
                sys.exit(1)
            create_directories()
            check_env_file()
            start_production()
            return
    
    # Default: start in development mode
    print_banner()
    
    # System checks
    check_python_version()
    if not check_dependencies():
        sys.exit(1)
    
    create_directories()
    check_env_file()
    
    # Start the application
    start_flask_app()

if __name__ == "__main__":
    main() 