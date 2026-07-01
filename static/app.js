// ─── STATE MANAGEMENT ───
const API_BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';
let liveLossChart = null;
let timeMemChart = null;
let paramsSizeChart = null;
let emotionScoreChart = null;
let eventSource = null;

// Theme colors for Chart.js
const colors = {
    lora: '#00bcff',
    loraGlow: 'rgba(0, 188, 255, 0.15)',
    full: '#ff455b',
    fullGlow: 'rgba(255, 69, 91, 0.15)',
    base: '#9ca3af',
    baseGlow: 'rgba(156, 163, 175, 0.15)',
    grid: 'rgba(255, 255, 255, 0.05)',
    text: '#9ca3af'
};

const modelChartProps = {
    base: { label: 'Base', borderColor: colors.base, backgroundColor: 'rgba(156, 163, 175, 0.55)' },
    lora: { label: 'LoRA', borderColor: colors.lora, backgroundColor: 'rgba(0, 188, 255, 0.55)' },
    full: { label: 'Full FT', borderColor: colors.full, backgroundColor: 'rgba(255, 69, 91, 0.55)' }
};

// ─── INITIALIZE ON LOAD ───
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadConfig();
    initLiveLossChart();
    initEmotionScoreChart();
    checkTrainingStatus();
    loadAnalytics();
    setupSSE();
    
    // Configuration form
    document.getElementById('config-form').addEventListener('submit', saveConfig);
    
    // Training buttons
    document.getElementById('start-lora-btn').addEventListener('click', () => startTraining('lora'));
    document.getElementById('start-full-btn').addEventListener('click', () => startTraining('full'));
    
    // Console
    document.getElementById('clear-console-btn').addEventListener('click', () => {
        document.getElementById('console-output').innerHTML = '';
    });
    
    // Playground
    document.getElementById('predict-btn').addEventListener('click', runPlaygroundInference);
});

// ─── TABS LOGIC ───
function initTabs() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');
            
            // Re-render charts if needed to avoid size issues
            if (targetTab === 'analytics') {
                loadAnalytics();
            }
            if (targetTab === 'playground' && emotionScoreChart) {
                emotionScoreChart.resize();
                emotionScoreChart.update();
            }
        });
    });
}

// ─── CONFIGURATION ENDPOINTS ───
async function loadConfig() {
    try {
        const res = await fetch(`${API_BASE}/api/config`);
        const config = await res.json();
        
        document.getElementById('train_samples').value = config.train_samples;
        document.getElementById('epochs').value = config.epochs;
        document.getElementById('batch_size').value = config.batch_size;
        document.getElementById('learning_rate_lora').value = config.learning_rate_lora;
        document.getElementById('learning_rate_full').value = config.learning_rate_full;
        document.getElementById('lora_r').value = config.lora_r;
        document.getElementById('lora_alpha').value = config.lora_alpha;
        
        writeToConsole("[System] Current configuration loaded successfully.");
    } catch (err) {
        writeToConsole("[Error] Failed to load configuration: " + err.message, "error");
    }
}

async function saveConfig(e) {
    e.preventDefault();
    const config = {
        train_samples: parseInt(document.getElementById('train_samples').value),
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        learning_rate_lora: parseFloat(document.getElementById('learning_rate_lora').value),
        learning_rate_full: parseFloat(document.getElementById('learning_rate_full').value),
        lora_r: parseInt(document.getElementById('lora_r').value),
        lora_alpha: parseInt(document.getElementById('lora_alpha').value)
    };

    try {
        const res = await fetch(`${API_BASE}/api/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (res.ok) {
            writeToConsole("[System] Configuration saved on server.");
            alert("Configuration saved!");
        } else {
            writeToConsole("[Error] Server rejected configuration: " + data.detail, "error");
        }
    } catch (err) {
        writeToConsole("[Error] Failed to save configuration: " + err.message, "error");
    }
}

// ─── TRAINING CONTROL ───
async function startTraining(method) {
    try {
        const res = await fetch(`/api/train/${method}`, { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            writeToConsole(`[Training] Starting session for: ${method.toUpperCase()}...`);
            setTrainingUI(true, method);
            resetLiveChartForMethod(method);
        } else {
            writeToConsole(`[Error] Could not start training: ${data.detail}`, "error");
        }
    } catch (err) {
        writeToConsole(`[Error] Connection failed: ${err.message}`, "error");
    }
}

async function checkTrainingStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const status = await res.json();
        if (status.is_training) {
            setTrainingUI(true, status.current_method);
            if (status.progress && status.progress.progress_pct !== undefined) {
                updateProgressUI(status.current_method, status.progress);
            }
        } else {
            setTrainingUI(false);
        }
    } catch (err) {
        console.error("Error checking status:", err);
    }
}

function setTrainingUI(isTraining, method = null) {
    const loraBtn = document.getElementById('start-lora-btn');
    const fullBtn = document.getElementById('start-full-btn');
    const saveBtn = document.getElementById('save-config-btn');
    const progressSec = document.getElementById('live-progress-sec');
    const statusText = document.querySelector('#system-status .status-text');
    const statusDot = document.querySelector('#system-status .status-dot');

    if (isTraining) {
        loraBtn.disabled = true;
        fullBtn.disabled = true;
        saveBtn.disabled = true;
        progressSec.classList.remove('hidden');
        
        const methodTitle = method === 'lora' ? 'LoRA Fine-Tuning' : 'Full Fine-Tuning';
        document.getElementById('progress-title').innerText = `Training ${methodTitle} in progress...`;
        
        statusText.innerText = `Training ${method === 'lora' ? 'LoRA' : 'Full FT'}...`;
        statusDot.className = `status-dot pulsing-${method === 'lora' ? 'blue' : 'red'}`;
    } else {
        loraBtn.disabled = false;
        fullBtn.disabled = false;
        saveBtn.disabled = false;
        progressSec.classList.add('hidden');
        
        statusText.innerText = "System Ready";
        statusDot.className = "status-dot green";
    }
}

function updateProgressUI(method, progress) {
    const pct = progress.progress_pct || 0;
    document.getElementById('progress-val').innerText = `${pct}%`;
    document.getElementById('progress-bar-fill').style.width = `${pct}%`;
    document.getElementById('progress-bar-fill').style.background = 
        method === 'lora' 
            ? `linear-gradient(90deg, ${colors.lora}, #a855f7)`
            : `linear-gradient(90deg, ${colors.full}, #ff7a00)`;

    document.getElementById('stat-epoch').innerText = progress.epoch !== undefined ? progress.epoch : '-';
    document.getElementById('stat-step').innerText = progress.step !== undefined ? progress.step : '-';
    document.getElementById('stat-max-step').innerText = progress.max_steps !== undefined ? progress.max_steps : '-';
    document.getElementById('stat-loss').innerText = progress.loss !== null && progress.loss !== undefined ? progress.loss.toFixed(4) : '-';
}

// ─── SSE CONNECTION (LIVE LOGS & CHARTS) ───
function setupSSE() {
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource(`${API_BASE}/api/stream-progress`);
    
    eventSource.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'status') {
            if (msg.status === 'started') {
                writeToConsole(`[Server] ${msg.method.toUpperCase()} training started.`, msg.method + "-color");
                setTrainingUI(true, msg.method);
            } else if (msg.status === 'completed') {
                writeToConsole(`[Server] ${msg.method.toUpperCase()} training completed successfully!`, "system");
                setTrainingUI(false);
                loadAnalytics(); // Reload analytics with fresh data
            } else if (msg.status === 'failed') {
                writeToConsole(`[Server Error] ${msg.method.toUpperCase()} training failed: ${msg.error}`, "error");
                setTrainingUI(false);
            }
        } 
        
        else if (msg.type === 'progress') {
            updateProgressUI(msg.method, msg.data);
            
            // Add progress logs to the console
            const stepStr = `[Step ${msg.data.step}/${msg.data.max_steps}]`;
            const lossStr = msg.data.loss !== null ? `Loss: ${msg.data.loss.toFixed(4)}` : '';
            const evalLossStr = msg.data.eval_loss !== null ? `Eval Loss: ${msg.data.eval_loss.toFixed(4)}` : '';
            
            if (lossStr || evalLossStr) {
                writeToConsole(`${stepStr} ${lossStr} ${evalLossStr}`, msg.method + "-color");
            }
            
            // Add points to the live loss chart
            if (msg.data.loss !== null) {
                addLiveLossPoint(msg.method, msg.data.step, msg.data.loss);
            }
        }
    };
    
    eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        writeToConsole("[System] Progress stream connection lost. Attempting to reconnect...", "error");
    };
}

// ─── CONSOLE UTILS ───
function writeToConsole(text, type = "") {
    const consoleBody = document.getElementById('console-output');
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    line.innerText = `[${timestamp}] ${text}`;
    
    consoleBody.appendChild(line);
    consoleBody.scrollTop = consoleBody.scrollHeight;
}

// ─── LIVE LOSS CHART (CHART.JS) ───
function initLiveLossChart() {
    const ctx = document.getElementById('liveLossChart').getContext('2d');
    liveLossChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'LoRA Loss',
                    data: [],
                    borderColor: colors.lora,
                    backgroundColor: colors.loraGlow,
                    borderWidth: 2,
                    tension: 0.2,
                    pointRadius: 2,
                    fill: true
                },
                {
                    label: 'Full FT Loss',
                    data: [],
                    borderColor: colors.full,
                    backgroundColor: colors.fullGlow,
                    borderWidth: 2,
                    tension: 0.2,
                    pointRadius: 2,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Training Steps', color: colors.text },
                    grid: { color: colors.grid },
                    ticks: { color: colors.text }
                },
                y: {
                    title: { display: true, text: 'Loss Value', color: colors.text },
                    grid: { color: colors.grid },
                    ticks: { color: colors.text }
                }
            },
            plugins: {
                legend: { labels: { color: colors.text } }
            }
        }
    });
}

function initEmotionScoreChart() {
    const ctx = document.getElementById('emotionScoreChart').getContext('2d');
    emotionScoreChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Emotion', color: colors.text },
                    grid: { color: colors.grid },
                    ticks: { color: colors.text }
                },
                y: {
                    title: { display: true, text: 'Probability', color: colors.text },
                    min: 0,
                    max: 1,
                    grid: { color: colors.grid },
                    ticks: {
                        color: colors.text,
                        callback: (value) => `${value * 100}%`
                    }
                }
            },
            plugins: {
                legend: { labels: { color: colors.text } },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(1)}%`
                    }
                }
            }
        }
    });
}

function updateEmotionScoreChart(scoreData) {
    const availableScores = Object.values(scoreData).find(scores => scores && Object.keys(scores).length > 0);
    const labels = availableScores ? Object.keys(availableScores) : [];
    emotionScoreChart.data.labels = labels;
    emotionScoreChart.data.datasets = [];

    Object.entries(scoreData).forEach(([key, scores]) => {
        if (!scores || Object.keys(scores).length === 0) {
            return;
        }
        const props = modelChartProps[key];
        emotionScoreChart.data.datasets.push({
            label: props.label,
            data: labels.map(label => scores[label] ?? 0),
            backgroundColor: props.backgroundColor,
            borderColor: props.borderColor,
            borderWidth: 1,
            borderRadius: 6,
            categoryPercentage: 0.7,
            barPercentage: 0.8
        });
    });

    emotionScoreChart.update();
    emotionScoreChart.resize();
    document.getElementById('emotion-chart-legend').innerText = labels.length
        ? 'Higher bars indicate stronger predicted probability for the emotion on the same input text.'
        : 'Run inference to view the full emotion probability breakdown for each available model.';
}

function resetLiveChartForMethod(method) {
    const datasetIndex = method === 'lora' ? 0 : 1;
    liveLossChart.data.datasets[datasetIndex].data = [];
    liveLossChart.update();
}

function addLiveLossPoint(method, step, loss) {
    const datasetIndex = method === 'lora' ? 0 : 1;
    liveLossChart.data.datasets[datasetIndex].data.push({ x: step, y: loss });
    
    // Sort by step to keep the chart data ordered
    liveLossChart.data.datasets[datasetIndex].data.sort((a, b) => a.x - b.x);
    
    liveLossChart.update('none'); // Update silently without animation for speed
}

// ─── TAB 2: ANALYTICS & GREEN AI ───
async function loadAnalytics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        const metrics = await res.json();
        
        const hasLora = metrics.lora !== null;
        const hasFull = metrics.full !== null;
        
        const alertSec = document.getElementById('metrics-alert');
        const contentSec = document.getElementById('analytics-content');
        
        // Control overlay visibility in the playground tab
        document.getElementById('overlay-lora').style.display = hasLora ? 'none' : 'flex';
        document.getElementById('overlay-full').style.display = hasFull ? 'none' : 'flex';
        
        if (!hasLora || !hasFull) {
            alertSec.style.display = 'flex';
            contentSec.style.display = 'none';
            
            // If only one model is available, we can populate partial KPIs, but charts need both for a full comparison.
            return;
        }
        
        alertSec.style.display = 'none';
        contentSec.style.display = 'flex';
        
        // Populate KPIs
        document.getElementById('kpi-lora-acc').innerText = `${(metrics.lora.final_accuracy * 100).toFixed(2)}%`;
        document.getElementById('kpi-full-acc').innerText = `${(metrics.full.final_accuracy * 100).toFixed(2)}%`;
        
        const energySavings = ((metrics.full.estimated_energy_kwh - metrics.lora.estimated_energy_kwh) / metrics.full.estimated_energy_kwh * 100).toFixed(1);
        document.getElementById('kpi-energy-savings').innerText = `-${energySavings}%`;
        
        // Render comparative charts
        renderTimeMemChart(metrics.lora, metrics.full);
        renderParamsSizeChart(metrics.lora, metrics.full);
        
        // Populate Green AI
        document.getElementById('green-energy-lora').innerText = `${metrics.lora.estimated_energy_kwh.toFixed(6)} kWh`;
        document.getElementById('green-energy-full').innerText = `${metrics.full.estimated_energy_kwh.toFixed(6)} kWh`;
        
        const efficiencyRatio = (metrics.full.estimated_energy_kwh / metrics.lora.estimated_energy_kwh).toFixed(1);
        document.getElementById('green-efficiency-ratio').innerText = `de ${efficiencyRatio} ori mai eficient`;
        
        document.getElementById('green-co2-lora').innerText = `${metrics.lora.estimated_co2_g.toFixed(4)} g`;
        document.getElementById('green-co2-full').innerText = `${metrics.full.estimated_co2_g.toFixed(4)} g`;
        
        const co2Saved = (metrics.full.estimated_co2_g - metrics.lora.estimated_co2_g).toFixed(4);
        document.getElementById('green-co2-saved').innerText = `${co2Saved} g CO2`;
        
    } catch (err) {
        console.error("Error loading analytics:", err);
    }
}

function renderTimeMemChart(lora, full) {
    const ctx = document.getElementById('timeMemChart').getContext('2d');
    
    if (timeMemChart) {
        timeMemChart.destroy();
    }
    
    timeMemChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Training Time (sec)', 'Peak RAM (MB)'],
            datasets: [
                {
                    label: 'LoRA',
                    data: [lora.train_time_sec, lora.peak_ram_mb],
                    backgroundColor: colors.lora,
                    borderColor: 'transparent',
                    borderRadius: 6
                },
                {
                    label: 'Full Fine-Tuning',
                    data: [full.train_time_sec, full.peak_ram_mb],
                    backgroundColor: colors.full,
                    borderColor: 'transparent',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: colors.grid }, ticks: { color: colors.text } },
                y: { grid: { color: colors.grid }, ticks: { color: colors.text }, type: 'logarithmic' } // Log scale helps visualize large differences
            },
            plugins: {
                legend: { labels: { color: colors.text } }
            }
        }
    });
}

function renderParamsSizeChart(lora, full) {
    const ctx = document.getElementById('paramsSizeChart').getContext('2d');
    
    if (paramsSizeChart) {
        paramsSizeChart.destroy();
    }
    
    paramsSizeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Trained Parameters (Million)', 'Model/Checkpoint Size (MB)'],
            datasets: [
                {
                    label: 'LoRA',
                    data: [lora.trainable_params / 1000000, lora.checkpoint_size_mb],
                    backgroundColor: colors.lora,
                    borderColor: 'transparent',
                    borderRadius: 6
                },
                {
                    label: 'Full Fine-Tuning',
                    data: [full.trainable_params / 1000000, full.checkpoint_size_mb],
                    backgroundColor: colors.full,
                    borderColor: 'transparent',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: colors.grid }, ticks: { color: colors.text } },
                y: { grid: { color: colors.grid }, ticks: { color: colors.text }, type: 'logarithmic' }
            },
            plugins: {
                legend: { labels: { color: colors.text } }
            }
        }
    });
}

// ─── TAB 3: INTERACTIVE PLAYGROUND ───
async function runPlaygroundInference() {
    const text = document.getElementById('playground-text').value.trim();
    if (!text) {
        alert("Please enter text for analysis.");
        return;
    }

    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

    // Reset the prediction UI and show loading state
    const models = ['base', 'lora', 'full'];
    models.forEach(m => {
        document.getElementById(`res-${m}-placeholder`).style.display = 'none';
        document.getElementById(`res-${m}-content`).classList.remove('hidden');
        document.getElementById(`res-${m}-label`).innerText = "Processing...";
        document.getElementById(`res-${m}-val`).innerText = "...";
        document.getElementById(`res-${m}-bar`).style.width = '0%';
    });

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        
        const scorePayload = { base: null, lora: null, full: null };

        models.forEach(m => {
            const mData = data[m];
            const contentDiv = document.getElementById(`res-${m}-content`);
            const labelDiv = document.getElementById(`res-${m}-label`);
            const valSpan = document.getElementById(`res-${m}-val`);
            const barDiv = document.getElementById(`res-${m}-bar`);
            const iconDiv = document.getElementById(`res-${m}-icon`);
            
            if (mData && mData.available) {
                // Set the label text for the prediction
                const labelLower = mData.label.toLowerCase();
                const labelUpper = mData.label.toUpperCase();
                labelDiv.innerText = labelUpper;
                labelDiv.className = `sentiment-label ${labelLower}`;
                
                // Set the confidence value and bar
                const confPct = Math.round(mData.confidence * 100);
                valSpan.innerText = `${confPct}%`;
                barDiv.style.width = `${confPct}%`;
                
                // Choose a more appropriate icon for the emotion label
                const emotionIcon = {
                    sadness: 'fa-face-frown text-red',
                    joy: 'fa-face-laugh text-yellow',
                    love: 'fa-heart text-pink',
                    anger: 'fa-face-angry text-red',
                    fear: 'fa-face-grimace text-orange',
                    surprise: 'fa-face-surprise text-blue'
                }[labelLower] || 'fa-face-meh text-gray';
                iconDiv.innerHTML = `<i class="fa-solid ${emotionIcon}"></i>`;

                scorePayload[m] = mData.scores || null;
            } else {
                // If the model is unavailable, hide the content and show the overlay
                contentDiv.classList.add('hidden');
                document.getElementById(`res-${m}-placeholder`).style.display = 'block';
                document.getElementById(`res-${m}-placeholder`).innerText = "Model unavailable";
            }
        });

        updateEmotionScoreChart(scorePayload);
        
    } catch (err) {
        console.error("Error running inference:", err);
        alert("An error occurred while communicating with the server.");
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Emotion';
        if (emotionScoreChart) {
            emotionScoreChart.resize();
        }
    }
}
