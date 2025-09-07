#!/usr/bin/env python3
"""
Script to restart Streamlit and clear all cache.
Use this if you encounter caching issues with the streaming functionality.
"""

import subprocess
import sys
import time
import os

def restart_streamlit():
    """Restart Streamlit app with cache cleared."""
    print("🔄 Restarting Streamlit with fresh cache...")
    
    # Kill any existing streamlit processes
    try:
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        time.sleep(2)
    except:
        pass
    
    # Start fresh streamlit instance
    try:
        print("🚀 Starting Streamlit app...")
        env = os.environ.copy()
        # Clear any streamlit cache environment variables
        env.pop('STREAMLIT_SERVER_ENABLE_STATIC_SERVING', None)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], env=env)
    except KeyboardInterrupt:
        print("\n👋 Streamlit stopped.")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

if __name__ == "__main__":
    restart_streamlit()
