# 🔄 Streaming Progress Troubleshooting Guide

## Issue: 'APIClient' object has no attribute 'conduct_research_stream'

This error occurs when Streamlit's cache contains an old version of the `APIClient` class. Here are several solutions:

### 🔧 **Quick Fixes**

#### Option 1: Use Debug Mode
1. Go to the "AI Research" page
2. Enable "🔧 Debug Mode" checkbox
3. Check if "Has streaming method" shows `True`
4. If `False`, click "Clear Cache" button

#### Option 2: Restart Streamlit
```bash
# Kill current streamlit process
pkill -f streamlit

# Start fresh
python restart_streamlit.py
# OR
streamlit run streamlit_app.py
```

#### Option 3: Manual Cache Clear
In your terminal:
```bash
# Clear streamlit cache directory
rm -rf ~/.streamlit/cache/
```

### 🚀 **Verification Steps**

#### Test API Endpoint Directly
```bash
# Test if the streaming endpoint works
curl -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main risks companies face?"}'
```

#### Test Streaming Function
```bash
python test_streaming.py
```

#### Check API Server
```bash
# Make sure API server is running
python api.py
```

### 🔍 **Debug Information**

#### Check Method Availability
```python
from streamlit_app import APIClient
client = APIClient()
print(hasattr(client, 'conduct_research_stream'))
print(dir(client))
```

#### Verify API Response
```python
import requests
response = requests.post("http://localhost:8000/research/stream", 
                        json={"question": "test"}, stream=True)
print(response.status_code)
for line in response.iter_lines():
    print(line)
    break
```

### 📋 **Implementation Details**

The streaming implementation includes:

1. **Backend**: `/research/stream` endpoint in `api.py`
2. **Agent**: `conduct_10k_research_with_progress()` in `tenk_research_agent.py` 
3. **Frontend**: `conduct_research_stream()` method in `streamlit_app.py`

### 🛠️ **Manual Fix**

If issues persist, manually refresh the API client:

```python
# In streamlit_app.py, replace the cached call with:
api_client = APIClient()  # Direct instantiation
```

### 📞 **Still Having Issues?**

1. Check that all files were saved properly
2. Restart your IDE/editor
3. Restart the Python interpreter
4. Check for any import errors in the console
5. Use the fallback mechanism in the code that creates a fresh APIClient instance

### 🎯 **Expected Behavior**

When working correctly, you should see:
- Real-time progress bar updates
- Step-by-step status messages  
- Expandable research details
- Smooth streaming from 0% to 100%

The debug mode will show:
- `Has streaming method: True`
- List of available methods including `conduct_research_stream`
