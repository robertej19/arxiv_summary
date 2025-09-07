# Frontend Integration for Fast 10-K Research

The fast research agents are now fully integrated into the FastAPI backend and accessible from any frontend application.

## 🚀 Quick Start

### 1. Start the API Server
```bash
cd /home/rober/arxiv_summarizer/fast_run
python -m uvicorn api:app --reload
```

### 2. View the Demo
Open `demo_frontend.html` in your browser to see the interactive research widget.

### 3. Test API Endpoints
```bash
# Get available research modes
curl http://localhost:8000/research/modes

# Fast research
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are AI risks?", "mode": "fast"}'

# Ultra-fast research  
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are AI risks?", "mode": "ultra_fast"}'
```

## 📡 API Endpoints

### Research Modes
- **GET** `/research/modes` - Get available research modes and capabilities

### Conduct Research  
- **POST** `/research` - Main research endpoint with mode selection

**Request Body:**
```json
{
  "question": "Your research question",
  "mode": "ultra_fast" | "fast" | "deep"
}
```

**Response:**
```json
{
  "question": "Your research question",
  "answer": "Research findings...",
  "status": "completed",
  "mode": "fast", 
  "processing_time": 1.234,
  "cached": false
}
```

### Streaming Research (Deep Mode)
- **POST** `/research/stream` - Streaming research with progress updates

## ⚡ Research Modes

| Mode | Response Time | Best For | Limitations |
|------|---------------|----------|-------------|
| **ultra_fast** | < 0.1s | Common topics (AI, cybersecurity, supply chain) | Pre-defined templates only |
| **fast** | 1-3s | Most questions, custom analysis | Shorter responses |
| **deep** | 30s+ | Comprehensive analysis, comparisons | Slower response time |

## 🔧 Frontend Integration

### JavaScript API Client

```javascript
// Initialize API client
const api = new TenKResearchAPI('http://localhost:8000');

// Get available modes
const modes = await api.getResearchModes();

// Conduct research
const result = await api.conductResearch(
  "What are AI risks?", 
  "ultra_fast"
);

// Streaming research (deep mode)
const answer = await api.streamResearch(
  "Complex research question",
  (progress) => {
    console.log(`${progress.step}: ${progress.message}`);
    // Update progress bar: progress.progress (0.0 to 1.0)
  }
);
```

### React Integration Example

```jsx
import { useState, useEffect } from 'react';

function ResearchComponent() {
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState('fast');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleResearch = async () => {
    setLoading(true);
    try {
      const api = new TenKResearchAPI();
      const response = await api.conductResearch(question, mode);
      setResult(response);
    } catch (error) {
      console.error('Research failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <select value={mode} onChange={(e) => setMode(e.target.value)}>
        <option value="ultra_fast">⚡ Ultra-Fast</option>
        <option value="fast">🚀 Fast</option>
        <option value="deep">🔍 Deep</option>
      </select>
      
      <textarea 
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Enter your research question"
      />
      
      <button onClick={handleResearch} disabled={loading}>
        {loading ? 'Researching...' : 'Research'}
      </button>

      {result && (
        <div>
          <h3>Results ({result.processing_time}s)</h3>
          <p>{result.answer}</p>
        </div>
      )}
    </div>
  );
}
```

### Vue Integration Example

```vue
<template>
  <div class="research-component">
    <select v-model="mode">
      <option value="ultra_fast">⚡ Ultra-Fast</option>
      <option value="fast">🚀 Fast</option>  
      <option value="deep">🔍 Deep</option>
    </select>

    <textarea 
      v-model="question"
      placeholder="Enter your research question"
    ></textarea>

    <button @click="handleResearch" :disabled="loading">
      {{ loading ? 'Researching...' : 'Research' }}
    </button>

    <div v-if="result" class="result">
      <h3>Results ({{ result.processing_time }}s)</h3>
      <p v-html="formatAnswer(result.answer)"></p>
    </div>
  </div>
</template>

<script>
import { TenKResearchAPI } from './api/research';

export default {
  data() {
    return {
      question: '',
      mode: 'fast',
      result: null,
      loading: false,
      api: new TenKResearchAPI()
    };
  },
  methods: {
    async handleResearch() {
      this.loading = true;
      try {
        this.result = await this.api.conductResearch(this.question, this.mode);
      } catch (error) {
        console.error('Research failed:', error);
      } finally {
        this.loading = false;
      }
    },
    formatAnswer(answer) {
      return answer
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[(\d+)\]/g, '<sup>[$1]</sup>');
    }
  }
};
</script>
```

## 🎨 UI Components

### Mode Selector Component
```javascript
function ModeSelector({ selectedMode, onModeChange, modes }) {
  return (
    <div className="mode-selector">
      {Object.entries(modes.modes).map(([key, mode]) => (
        <button
          key={key}
          className={`mode-btn ${selectedMode === key ? 'active' : ''}`}
          onClick={() => onModeChange(key)}
        >
          <div className="mode-icon">{getIcon(key)}</div>
          <div className="mode-name">{mode.name}</div>
          <div className="mode-time">{mode.response_time}</div>
        </button>
      ))}
    </div>
  );
}

function getIcon(mode) {
  return {
    'ultra_fast': '⚡',
    'fast': '🚀',
    'deep': '🔍'
  }[mode] || '📊';
}
```

### Progress Component (for Deep Mode)
```javascript
function ResearchProgress({ progress, message }) {
  return (
    <div className="research-progress">
      <div className="progress-bar">
        <div 
          className="progress-fill"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
      <div className="progress-message">{message}</div>
    </div>
  );
}
```

## 🔌 CORS Configuration

The API includes CORS middleware for frontend integration. For production, update the allowed origins:

```python
# In api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Performance Monitoring

Track research performance in your frontend:

```javascript
class ResearchAnalytics {
  constructor() {
    this.metrics = [];
  }

  trackResearch(mode, question, processingTime) {
    this.metrics.push({
      mode,
      question: question.substring(0, 50),
      processingTime,
      timestamp: Date.now()
    });

    // Log performance
    console.log(`Research completed in ${processingTime}s (${mode} mode)`);
    
    // Send to analytics service
    this.sendMetrics({
      mode,
      processing_time: processingTime,
      question_length: question.length
    });
  }

  getAverageTime(mode) {
    const modeMetrics = this.metrics.filter(m => m.mode === mode);
    if (modeMetrics.length === 0) return 0;
    
    const totalTime = modeMetrics.reduce((sum, m) => sum + m.processingTime, 0);
    return totalTime / modeMetrics.length;
  }
}
```

## 🚨 Error Handling

Handle different error scenarios:

```javascript
async function robustResearch(api, question, mode) {
  try {
    return await api.conductResearch(question, mode);
  } catch (error) {
    if (error.message.includes('Research mode')) {
      // Mode not available, fallback to fast mode
      console.warn(`Mode ${mode} not available, using fast mode`);
      return await api.conductResearch(question, 'fast');
    } else if (error.message.includes('timeout')) {
      // Timeout, suggest faster mode
      throw new Error('Request timed out. Try using fast or ultra-fast mode.');
    } else {
      // Other errors
      throw error;
    }
  }
}
```

## 🎯 Best Practices

### Mode Selection Logic
```javascript
function suggestMode(question, userPreference = 'fast') {
  // Ultra-fast mode for common topics
  const ultraFastKeywords = ['ai', 'artificial intelligence', 'cybersecurity', 'supply chain', 'climate'];
  const hasUltraFastKeyword = ultraFastKeywords.some(keyword => 
    question.toLowerCase().includes(keyword)
  );
  
  if (hasUltraFastKeyword && question.length < 100) {
    return 'ultra_fast';
  }
  
  // Deep mode for complex questions
  if (question.length > 200 || question.includes('compare') || question.includes('analyze')) {
    return 'deep';
  }
  
  // Default to fast mode
  return 'fast';
}
```

### Caching Strategy
```javascript
class ResearchCache {
  constructor(maxSize = 50) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  getCacheKey(question, mode) {
    return `${mode}:${question.toLowerCase().trim()}`;
  }

  get(question, mode) {
    const key = this.getCacheKey(question, mode);
    const cached = this.cache.get(key);
    
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes
      return cached.result;
    }
    
    return null;
  }

  set(question, mode, result) {
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    const key = this.getCacheKey(question, mode);
    this.cache.set(key, {
      result,
      timestamp: Date.now()
    });
  }
}
```

## 🔧 Development Setup

1. **Install Dependencies**: Make sure all Python dependencies are installed
2. **Start API Server**: `python -m uvicorn api:app --reload`
3. **Open Demo**: Open `demo_frontend.html` in browser
4. **Test Endpoints**: Use the provided curl commands or Postman

## 📱 Mobile Considerations

For mobile-responsive design:

```css
@media (max-width: 768px) {
  .research-widget {
    padding: 15px;
    margin: 10px;
  }
  
  .speed-comparison {
    grid-template-columns: 1fr;
    gap: 15px;
  }
  
  .question-input textarea {
    font-size: 16px; /* Prevents zoom on iOS */
  }
}
```

## 🎉 Success!

Your fast research system is now fully frontend-accessible with:

- ⚡ **3 speed modes** (ultra-fast, fast, deep)
- 🔄 **Streaming progress** for deep research  
- 📡 **RESTful API** with proper error handling
- 🎨 **Interactive demo** with sample questions
- 📚 **Complete integration examples** for popular frameworks

The frontend can now deliver research responses in **seconds instead of minutes** while maintaining the quality and accuracy of your 10-K analysis system!
