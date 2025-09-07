// Frontend Integration Example for Fast 10-K Research API
// This shows how to integrate the fast research modes into a frontend application

class TenKResearchAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    // Get available research modes
    async getResearchModes() {
        const response = await fetch(`${this.baseUrl}/research/modes`);
        return await response.json();
    }

    // Conduct research with specified mode
    async conductResearch(question, mode = 'fast') {
        const response = await fetch(`${this.baseUrl}/research`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                mode: mode
            })
        });
        
        if (!response.ok) {
            throw new Error(`Research failed: ${response.statusText}`);
        }
        
        return await response.json();
    }

    // Stream research with progress updates (deep mode only)
    async streamResearch(question, onProgress) {
        const response = await fetch(`${this.baseUrl}/research/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                mode: 'deep'  // Streaming only available for deep mode
            })
        });

        if (!response.ok) {
            throw new Error(`Stream research failed: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    onProgress(data);
                    
                    if (data.step === 'completed') {
                        return data.details.final_answer;
                    }
                }
            }
        }
    }
}

// Example usage in a React/Vue/vanilla JS frontend

class ResearchWidget {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.api = new TenKResearchAPI();
        this.init();
    }

    async init() {
        // Get available modes
        const modes = await this.api.getResearchModes();
        this.createUI(modes);
    }

    createUI(modes) {
        this.container.innerHTML = `
            <div class="research-widget">
                <h3>10-K Research Assistant</h3>
                
                <!-- Mode Selection -->
                <div class="mode-selection">
                    <label>Research Mode:</label>
                    <select id="research-mode">
                        <option value="ultra_fast">⚡ Ultra-Fast (${modes.modes.ultra_fast.response_time})</option>
                        <option value="fast" selected>🚀 Fast (${modes.modes.fast.response_time})</option>
                        <option value="deep">🔍 Deep (${modes.modes.deep.response_time})</option>
                    </select>
                </div>

                <!-- Question Input -->
                <div class="question-input">
                    <textarea id="research-question" 
                              placeholder="Ask about companies' AI risks, cybersecurity, supply chain, etc."
                              rows="3"></textarea>
                    <button id="research-btn">Research</button>
                </div>

                <!-- Mode Info -->
                <div id="mode-info" class="mode-info"></div>

                <!-- Results -->
                <div id="research-results" class="results"></div>
                
                <!-- Progress (for deep mode) -->
                <div id="progress" class="progress" style="display: none;"></div>
            </div>
        `;

        this.setupEventHandlers(modes);
    }

    setupEventHandlers(modes) {
        const modeSelect = document.getElementById('research-mode');
        const questionInput = document.getElementById('research-question');
        const researchBtn = document.getElementById('research-btn');
        const modeInfo = document.getElementById('mode-info');
        const results = document.getElementById('research-results');
        const progress = document.getElementById('progress');

        // Update mode info when selection changes
        modeSelect.addEventListener('change', (e) => {
            const selectedMode = modes.modes[e.target.value];
            modeInfo.innerHTML = `
                <strong>${selectedMode.name}</strong><br>
                ${selectedMode.description}<br>
                <em>Best for: ${selectedMode.best_for.join(', ')}</em>
            `;
        });

        // Trigger initial mode info
        modeSelect.dispatchEvent(new Event('change'));

        // Conduct research
        researchBtn.addEventListener('click', async () => {
            const question = questionInput.value.trim();
            const mode = modeSelect.value;

            if (!question) {
                alert('Please enter a research question');
                return;
            }

            researchBtn.disabled = true;
            researchBtn.textContent = 'Researching...';
            results.innerHTML = '';
            progress.style.display = 'none';

            try {
                if (mode === 'deep') {
                    // Use streaming for deep mode
                    progress.style.display = 'block';
                    
                    const answer = await this.api.streamResearch(question, (progressData) => {
                        progress.innerHTML = `
                            <div class="progress-bar">
                                <div style="width: ${progressData.progress * 100}%"></div>
                            </div>
                            <div class="progress-text">${progressData.message}</div>
                        `;
                    });

                    this.displayResults({
                        question,
                        answer,
                        mode,
                        status: 'completed'
                    });

                } else {
                    // Use regular API for fast modes
                    const result = await this.api.conductResearch(question, mode);
                    this.displayResults(result);
                }

            } catch (error) {
                results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                researchBtn.disabled = false;
                researchBtn.textContent = 'Research';
                progress.style.display = 'none';
            }
        });

        // Sample questions
        this.addSampleQuestions(questionInput);
    }

    displayResults(result) {
        const results = document.getElementById('research-results');
        
        const modeIcon = {
            'ultra_fast': '⚡',
            'fast': '🚀', 
            'deep': '🔍'
        }[result.mode] || '📊';

        results.innerHTML = `
            <div class="research-result">
                <div class="result-header">
                    <span class="mode-badge">${modeIcon} ${result.mode.replace('_', '-')}</span>
                    <span class="timing">⏱️ ${result.processing_time?.toFixed(3) || 0}s</span>
                </div>
                
                <div class="question">
                    <strong>Question:</strong> ${result.question}
                </div>
                
                <div class="answer">
                    <strong>Answer:</strong>
                    <div class="answer-content">${this.formatAnswer(result.answer)}</div>
                </div>
            </div>
        `;
    }

    formatAnswer(answer) {
        // Convert markdown-style formatting to HTML
        return answer
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\[(\d+)\]/g, '<sup class="citation">[$1]</sup>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
    }

    addSampleQuestions(questionInput) {
        const samples = [
            "What are AI risks?",
            "How do companies approach cybersecurity?",
            "What supply chain issues do companies face?",
            "What climate change risks do manufacturing companies identify?",
            "How are tech companies investing in AI?"
        ];

        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'sample-questions';
        sampleContainer.innerHTML = '<small>Sample questions:</small>';
        
        samples.forEach(sample => {
            const btn = document.createElement('button');
            btn.textContent = sample;
            btn.className = 'sample-btn';
            btn.onclick = () => questionInput.value = sample;
            sampleContainer.appendChild(btn);
        });

        questionInput.parentNode.appendChild(sampleContainer);
    }
}

// CSS for the widget (add to your stylesheet)
const widgetCSS = `
.research-widget {
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-family: Arial, sans-serif;
}

.mode-selection {
    margin-bottom: 15px;
}

.mode-selection select {
    padding: 8px;
    margin-left: 10px;
    border-radius: 4px;
    border: 1px solid #ccc;
}

.question-input {
    margin-bottom: 15px;
}

.question-input textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    resize: vertical;
}

.question-input button {
    margin-top: 10px;
    padding: 10px 20px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.question-input button:disabled {
    background: #ccc;
    cursor: not-allowed;
}

.mode-info {
    padding: 10px;
    background: #f8f9fa;
    border-radius: 4px;
    margin-bottom: 15px;
    font-size: 14px;
}

.sample-questions {
    margin-top: 10px;
}

.sample-btn {
    margin: 2px;
    padding: 4px 8px;
    background: #e9ecef;
    border: 1px solid #ccc;
    border-radius: 3px;
    cursor: pointer;
    font-size: 12px;
}

.sample-btn:hover {
    background: #dee2e6;
}

.research-result {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 15px;
    margin-top: 15px;
}

.result-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
}

.mode-badge {
    background: #28a745;
    color: white;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 12px;
    text-transform: uppercase;
}

.timing {
    font-size: 12px;
    color: #666;
}

.question {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
}

.answer-content {
    margin-top: 10px;
    line-height: 1.6;
}

.citation {
    color: #007bff;
    font-size: 11px;
}

.progress {
    margin: 15px 0;
}

.progress-bar {
    width: 100%;
    height: 20px;
    background: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
}

.progress-bar div {
    height: 100%;
    background: linear-gradient(90deg, #007bff, #28a745);
    transition: width 0.3s ease;
}

.progress-text {
    margin-top: 5px;
    font-size: 14px;
    color: #666;
}

.error {
    color: #dc3545;
    padding: 10px;
    background: #f8d7da;
    border-radius: 4px;
}
`;

// Initialize the widget when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Add CSS
    const style = document.createElement('style');
    style.textContent = widgetCSS;
    document.head.appendChild(style);
    
    // Initialize widget (assumes you have a div with id="research-widget")
    if (document.getElementById('research-widget')) {
        new ResearchWidget('research-widget');
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TenKResearchAPI, ResearchWidget };
}
