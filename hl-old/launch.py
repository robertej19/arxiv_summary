#!/usr/bin/env python3
"""
Simple launcher for the 10-K Knowledge Base web application.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if knowledge base exists."""
    if not Path("10k_knowledge_base.db").exists():
        print("❌ Error: Knowledge base not found!")
        print("Please run 'python build_10k_knowledge_base.py' first to create the database.")
        return False
    return True

def install_dependencies():
    """Install web dependencies."""
    print("📦 Installing web dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_api():
    """Start the FastAPI backend."""
    print("🔌 Starting FastAPI backend...")
    return subprocess.Popen([sys.executable, "api.py"])

def start_streamlit():
    """Start the Streamlit frontend."""
    print("📊 Starting Streamlit frontend...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def main():
    """Main launcher function."""
    print("🚀 10-K Knowledge Base Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    try:
        # Start services
        api_process = start_api()
        time.sleep(3)  # Wait for API to start
        
        streamlit_process = start_streamlit()
        time.sleep(2)  # Wait for Streamlit to start
        
        print("\n🎉 Application started successfully!")
        print("\n📊 Frontend (Streamlit): http://localhost:8501")
        print("🔌 Backend API (FastAPI): http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the application")
        
        # Open browser
        time.sleep(1)
        webbrowser.open("http://localhost:8501")
        
        # Wait for processes
        try:
            api_process.wait()
            streamlit_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            api_process.terminate()
            streamlit_process.terminate()
            api_process.wait()
            streamlit_process.wait()
            print("✅ Application stopped")
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
