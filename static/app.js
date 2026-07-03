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
    loadDatasetStats();
    loadDuplicateStats();

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
        document.getElementById('weight_decay_full').value = config.weight_decay_full;
        document.getElementById('warmup_ratio_full').value = config.warmup_ratio_full;
        document.getElementById('epochs_full').value = config.epochs_full;

        writeToConsole("[System] Current configuration loaded successfully.");
        updateTrainingEstimate();
        updateLRSchedulerChart();
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
        lora_alpha: parseInt(document.getElementById('lora_alpha').value),
        weight_decay_full: parseFloat(document.getElementById('weight_decay_full').value),
        warmup_ratio_full: parseFloat(document.getElementById('warmup_ratio_full').value),
        epochs_full: parseInt(document.getElementById('epochs_full').value)
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

        if (statusText) statusText.innerText = `Training ${method === 'lora' ? 'LoRA' : 'Full FT'}...`;
        if (statusDot) statusDot.className = `status-dot pulsing-${method === 'lora' ? 'blue' : 'red'}`;
    } else {
        loraBtn.disabled = false;
        fullBtn.disabled = false;
        saveBtn.disabled = false;
        progressSec.classList.add('hidden');

        if (statusText) statusText.innerText = "System Ready";
        if (statusDot) statusDot.className = "status-dot green";
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
            if (msg.data.eval_loss !== null && msg.data.eval_loss !== undefined) {
                addLiveEvalLossPoint(msg.method, msg.data.step, msg.data.eval_loss);
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
// ─── LR SCHEDULER PREVIEW ───
// Slide 21: "Learning rate — the most important setting"
let lrSchedulerChart = null;
const WARMUP_RATIO_UI = 0.1; // matches config.py WARMUP_RATIO

function computeLRSchedule(peakLR, totalSteps) {
    const warmupSteps = Math.max(1, Math.round(totalSteps * WARMUP_RATIO_UI));
    const points = [];
    const n = Math.max(totalSteps, 2);
    const step = Math.max(1, Math.round(n / 60)); // ~60 points for a smooth curve

    for (let s = 0; s <= n; s += step) {
        let lr;
        if (s < warmupSteps) {
            lr = peakLR * (s / warmupSteps);
        } else {
            const progress = (s - warmupSteps) / Math.max(1, (n - warmupSteps));
            lr = peakLR * 0.5 * (1 + Math.cos(Math.PI * progress));
        }
        points.push({ x: s, y: lr });
    }
    return points;
}

function initLRSchedulerChart() {
    const canvas = document.getElementById('lrSchedulerChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    lrSchedulerChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'LoRA LR',
                    data: [],
                    borderColor: colors.lora,
                    backgroundColor: colors.loraGlow,
                    borderWidth: 2,
                    tension: 0.15,
                    pointRadius: 0,
                    fill: true
                },
                {
                    label: 'Full FT LR',
                    data: [],
                    borderColor: colors.full,
                    backgroundColor: colors.fullGlow,
                    borderWidth: 2,
                    tension: 0.15,
                    pointRadius: 0,
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
                    title: { display: true, text: 'Learning Rate', color: colors.text },
                    grid: { color: colors.grid },
                    ticks: {
                        color: colors.text,
                        callback: v => v.toExponential(0)
                    }
                }
            },
            plugins: {
                legend: { labels: { color: colors.text } },
                tooltip: {
                    callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toExponential(3)}` }
                }
            }
        }
    });
    updateLRSchedulerChart();
}

function updateLRSchedulerChart() {
    if (!lrSchedulerChart) return;

    const samples = parseInt(document.getElementById('train_samples')?.value) || 200;
    const epochsLora = parseInt(document.getElementById('epochs')?.value) || 1;
    const epochsFull = parseInt(document.getElementById('epochs_full')?.value) || 1;
    const batch   = parseInt(document.getElementById('batch_size')?.value) || 8;
    const lrLora  = parseFloat(document.getElementById('learning_rate_lora')?.value) || 3e-4;
    const lrFull  = parseFloat(document.getElementById('learning_rate_full')?.value) || 5e-5;

    const stepsPerEpoch = Math.ceil(samples / batch);
    const totalStepsLora = Math.max(2, stepsPerEpoch * epochsLora);
    const totalStepsFull = Math.max(2, stepsPerEpoch * epochsFull);

    lrSchedulerChart.data.datasets[0].data = computeLRSchedule(lrLora, totalStepsLora);
    lrSchedulerChart.data.datasets[1].data = computeLRSchedule(lrFull, totalStepsFull);
    lrSchedulerChart.update('none');
}

function initLiveLossChart() {
    const ctx = document.getElementById('liveLossChart').getContext('2d');
    liveLossChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                // idx 0: LoRA train loss
                {
                    label: 'LoRA Train Loss',
                    data: [],
                    borderColor: colors.lora,
                    backgroundColor: colors.loraGlow,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 2,
                    fill: false
                },
                // idx 1: Full FT train loss
                {
                    label: 'Full FT Train Loss',
                    data: [],
                    borderColor: colors.full,
                    backgroundColor: colors.fullGlow,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 2,
                    fill: false
                },
                // idx 2: LoRA eval loss (dashed)
                {
                    label: 'LoRA Eval Loss',
                    data: [],
                    borderColor: colors.lora,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [6, 3],
                    tension: 0.3,
                    pointRadius: 4,
                    pointStyle: 'circle',
                    fill: false
                },
                // idx 3: Full FT eval loss (dashed)
                {
                    label: 'Full FT Eval Loss',
                    data: [],
                    borderColor: colors.full,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [6, 3],
                    tension: 0.3,
                    pointRadius: 4,
                    pointStyle: 'circle',
                    fill: false
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
                legend: {
                    labels: {
                        color: colors.text,
                        generateLabels: (chart) => {
                            const defaults = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                            defaults[2].lineDash = [6, 3];
                            defaults[3].lineDash = [6, 3];
                            return defaults;
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
                    }
                }
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
    const trainIdx = method === 'lora' ? 0 : 1;
    const evalIdx = method === 'lora' ? 2 : 3;
    liveLossChart.data.datasets[trainIdx].data = [];
    liveLossChart.data.datasets[evalIdx].data = [];
    liveLossChart.update();
    hideOverfitWarning();
}

function addLiveLossPoint(method, step, loss) {
    const datasetIndex = method === 'lora' ? 0 : 1;
    liveLossChart.data.datasets[datasetIndex].data.push({ x: step, y: loss });

    // Sort by step to keep the chart data ordered
    liveLossChart.data.datasets[datasetIndex].data.sort((a, b) => a.x - b.x);

    liveLossChart.update('none'); // Update silently without animation for speed
}

function addLiveEvalLossPoint(method, step, evalLoss) {
    const evalIdx = method === 'lora' ? 2 : 3;
    liveLossChart.data.datasets[evalIdx].data.push({ x: step, y: evalLoss });
    liveLossChart.data.datasets[evalIdx].data.sort((a, b) => a.x - b.x);
    liveLossChart.update('none');
    checkOverfitting(method);
}

// ─── OVERFITTING DETECTOR ───
// Slide 23: "Overfitting — the model memorizes instead of generalizing"
function checkOverfitting(method) {
    const trainIdx = method === 'lora' ? 0 : 1;
    const evalIdx = method === 'lora' ? 2 : 3;
    const trainData = liveLossChart.data.datasets[trainIdx].data;
    const evalData = liveLossChart.data.datasets[evalIdx].data;

    if (evalData.length < 2) return;

    const lastEval = evalData[evalData.length - 1].y;
    const prevEval = evalData[evalData.length - 2].y;
    const lastTrain = trainData.length ? trainData[trainData.length - 1].y : null;

    const evalRising = lastEval > prevEval;
    const gapWidening = lastTrain !== null && (lastEval - lastTrain) > 0.3;

    if (evalRising || gapWidening) {
        showOverfitWarning(method, evalRising, gapWidening);
    }
}

function showOverfitWarning(method, evalRising, gapWidening) {
    let box = document.getElementById('overfit-warning-box');
    if (!box) {
        box = document.createElement('div');
        box.id = 'overfit-warning-box';
        box.className = 'estimate-warning margin-top';
        document.getElementById('liveLossChart').closest('.card').appendChild(box);
    }
    const methodLabel = method === 'lora' ? 'LoRA' : 'Full FT';
    const reason = evalRising
        ? 'eval loss is rising compared to the previous step'
        : 'eval loss is diverging significantly from train loss';
    box.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i>
        <span><strong>${methodLabel}: possible overfitting</strong> — ${reason}. The model is starting to memorize the training data instead of generalizing (see "Common Pitfalls" slide). Consider fewer epochs or early stopping.</span>`;
    box.classList.remove('hidden');
}

function hideOverfitWarning() {
    const box = document.getElementById('overfit-warning-box');
    if (box) box.classList.add('hidden');
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
        document.getElementById('green-efficiency-ratio').innerText = `${efficiencyRatio}x more efficient`;
        
        document.getElementById('green-co2-lora').innerText = `${metrics.lora.estimated_co2_g.toFixed(4)} g`;
        document.getElementById('green-co2-full').innerText = `${metrics.full.estimated_co2_g.toFixed(4)} g`;
        
        const co2Saved = (metrics.full.estimated_co2_g - metrics.lora.estimated_co2_g).toFixed(4);
        document.getElementById('green-co2-saved').innerText = `${co2Saved} g CO2`;

        renderGreenEquivalents(metrics.lora, metrics.full);

        // Confusion matrix / per-class / misclassified
        window._analyticsMetrics = metrics;
        renderConfusionSection(metrics, currentConfusionMethod);

    } catch (err) {
        console.error("Error loading analytics:", err);
    }
}

// ─── GREEN AI: REAL-WORLD EQUIVALENTS ────────────────────────────────────────
function renderGreenEquivalents(lora, full) {
    const container = document.getElementById('green-equivalents');
    if (!container) return;

    const savedKwh = full.estimated_energy_kwh - lora.estimated_energy_kwh;
    const savedCo2 = full.estimated_co2_g - lora.estimated_co2_g;

    // Rough real-world reference points (order-of-magnitude, illustrative)
    const streamingMinutes = savedKwh / 0.0002; // ~0.0002 kWh/min for HD video streaming
    const phoneCharges = savedKwh / 0.012;       // ~12 Wh per full smartphone charge
    const ledMinutes = savedKwh / 0.00001667;    // ~10W LED bulb

    const equivalents = [
        { icon: '📺', value: streamingMinutes, unit: 'min of HD video streaming', label: 'Energy saved by using LoRA instead of Full FT' },
        { icon: '🔋', value: phoneCharges, unit: 'smartphone charges', label: 'Equivalent energy saved' },
        { icon: '💡', value: ledMinutes, unit: 'min of a 10W LED bulb', label: 'Equivalent energy saved' },
    ];

    container.innerHTML = equivalents.map(e => `
        <div class="green-equiv-card">
            <div class="green-equiv-icon">${e.icon}</div>
            <div class="green-equiv-text">
                <strong>${e.value < 0.01 ? '<0.01' : e.value < 1 ? e.value.toFixed(2) : Math.round(e.value).toLocaleString()} ${e.unit}</strong>
                <span>${e.label}</span>
            </div>
        </div>
    `).join('');
}

// ─── CONFUSION MATRIX, PER-CLASS METRICS, MISCLASSIFIED EXAMPLES ────────────
let currentConfusionMethod = 'lora';

document.addEventListener('DOMContentLoaded', () => {
    ['lora', 'full'].forEach(method => {
        const btn = document.getElementById(`cm-toggle-${method}`);
        if (btn) {
            btn.addEventListener('click', () => {
                currentConfusionMethod = method;
                document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                if (window._analyticsMetrics) {
                    renderConfusionSection(window._analyticsMetrics, method);
                }
            });
        }
    });
});

function renderConfusionSection(metrics, method) {
    const m = metrics[method];
    if (!m || !m.confusion_matrix) return;

    renderConfusionMatrix(m.confusion_matrix);
    renderPerClassTable(m.per_class, m.final_macro_f1);
    renderMisclassifiedTable(m.misclassified || []);
}

function renderConfusionMatrix(confusionMatrix) {
    const container = document.getElementById('confusion-matrix-container');
    if (!container) return;

    const labels = Object.keys(confusionMatrix);
    // Find max value for a simple heat intensity scale
    let maxVal = 0;
    labels.forEach(t => labels.forEach(p => { maxVal = Math.max(maxVal, confusionMatrix[t][p] || 0); }));

    let html = '<table class="cm-table"><thead><tr><th class="cm-axis-label"></th>';
    labels.forEach(p => { html += `<th>${p.slice(0, 4)}</th>`; });
    html += '</tr></thead><tbody>';

    labels.forEach(t => {
        html += `<tr><th>${t.slice(0, 4)}</th>`;
        labels.forEach(p => {
            const val = confusionMatrix[t][p] || 0;
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const isDiag = t === p;
            const bg = isDiag
                ? `rgba(16, 185, 129, ${0.15 + intensity * 0.5})`
                : `rgba(255, 69, 91, ${intensity * 0.45})`;
            html += `<td class="cm-cell" style="background:${bg}">${val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    html += '<div class="cm-caption">Rows = true label, columns = predicted label. Diagonal (green) = correct predictions.</div>';

    container.innerHTML = html;
}

function renderPerClassTable(perClass, macroF1) {
    const container = document.getElementById('per-class-table-container');
    if (!container || !perClass) return;

    let html = `<table class="pc-table"><thead><tr>
        <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th>
    </tr></thead><tbody>`;

    Object.entries(perClass).forEach(([label, s]) => {
        html += `<tr>
            <td>${label}</td>
            <td class="num">${(s.precision * 100).toFixed(1)}%</td>
            <td class="num">${(s.recall * 100).toFixed(1)}%</td>
            <td class="num">${(s.f1 * 100).toFixed(1)}%</td>
        </tr>`;
    });

    html += `<tr style="font-weight:700">
        <td>Macro Avg</td><td class="num">—</td><td class="num">—</td>
        <td class="num">${(macroF1 * 100).toFixed(1)}%</td>
    </tr>`;
    html += '</tbody></table>';

    container.innerHTML = html;
}

function renderMisclassifiedTable(misclassified) {
    const container = document.getElementById('misclassified-table');
    if (!container) return;

    if (!misclassified.length) {
        container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem">No misclassified examples recorded (or the model got everything right).</p>';
        return;
    }

    container.innerHTML = misclassified.map(r => `
        <div class="mis-row">
            <div class="mis-text">"${r.text}"</div>
            <div class="mis-labels">
                <span class="mis-true">true: ${r.true}</span>
                <span class="mis-pred">pred: ${r.pred}</span>
                <span class="mis-conf">conf ${(r.confidence * 100).toFixed(0)}%</span>
            </div>
        </div>
    `).join('');
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

// ─── DATASET EXPLORER ────────────────────────────────────────────────────────

const LABEL_COLORS = [
    '#00bcff', '#ff455b', '#10b981', '#f59e0b', '#a855f7', '#f97316'
];

let dsLabelChart = null;
let dsLenChart = null;

async function loadDatasetStats() {
    try {
        const res = await fetch(`${API_BASE}/api/dataset-stats`);
        if (!res.ok) return;
        const data = await res.json();
        renderDatasetStats(data);
    } catch (e) {
        console.warn('Dataset stats unavailable:', e);
    }
}

function renderDatasetStats(data) {
    // Header info
    document.getElementById('ds-name').textContent = data.dataset;
    document.getElementById('ds-task').textContent = data.task;

    // Split stat cards
    const splitContainer = document.getElementById('ds-split-cards');
    splitContainer.innerHTML = '';
    const splitOrder = ['train', 'validation', 'test'];
    const splitIcons = { train: '🟦', validation: '🟨', test: '🟩' };
    splitOrder.forEach(name => {
        const s = data.splits[name];
        if (!s) return;
        const card = document.createElement('div');
        card.className = 'ds-split-stat';
        card.innerHTML = `
            <div class="split-name">${splitIcons[name] || ''} ${name}</div>
            <div class="split-total">${s.total.toLocaleString()}</div>
            <div class="split-meta">avg ${s.avg_words} words &nbsp;·&nbsp; ${s.min_words}–${s.max_words} range</div>
        `;
        splitContainer.appendChild(card);
    });

    // Label distribution doughnut (train)
    const trainSplit = data.splits.train;
    if (trainSplit) {
        const labelNames = data.label_names;
        const counts = labelNames.map(l => trainSplit.label_counts[l] || 0);

        const ctxLabel = document.getElementById('ds-label-chart').getContext('2d');
        if (dsLabelChart) dsLabelChart.destroy();
        dsLabelChart = new Chart(ctxLabel, {
            type: 'doughnut',
            data: {
                labels: labelNames,
                datasets: [{
                    data: counts,
                    backgroundColor: LABEL_COLORS.slice(0, labelNames.length),
                    borderColor: 'rgba(0,0,0,0.3)',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#9ca3af', font: { size: 12 }, padding: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                                const pct = ((ctx.parsed / total) * 100).toFixed(1);
                                return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${pct}%)`;
                            }
                        }
                    }
                }
            }
        });

        // Text length bar chart (avg per label)
        const ctxLen = document.getElementById('ds-len-chart').getContext('2d');
        if (dsLenChart) dsLenChart.destroy();
        dsLenChart = new Chart(ctxLen, {
            type: 'bar',
            data: {
                labels: labelNames,
                datasets: [{
                    label: 'Count',
                    data: counts,
                    backgroundColor: LABEL_COLORS.slice(0, labelNames.length).map(c => c + 'bb'),
                    borderColor: LABEL_COLORS.slice(0, labelNames.length),
                    borderWidth: 1.5,
                    borderRadius: 6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                }
            }
        });
    }

    // Split comparison grouped bar chart (Train vs Val vs Test)
    const splitCanvas = document.getElementById('ds-split-compare-chart');
    if (splitCanvas) {
        const labelNames2 = data.label_names;
        const splitColors = { train: '#00bcff', validation: '#f59e0b', test: '#10b981' };
        const splitOrder = ['train', 'validation', 'test'];
        const splitLabels = { train: 'Train', validation: 'Validation', test: 'Test' };

        const datasets = splitOrder
            .filter(s => data.splits[s])
            .map(s => {
                const total = data.splits[s].total;
                const counts = labelNames2.map(l => {
                    const raw = data.splits[s].label_counts[l] || 0;
                    return total > 0 ? Math.round((raw / total) * 100 * 10) / 10 : 0;
                });
                return {
                    label: splitLabels[s],
                    data: counts,
                    backgroundColor: splitColors[s] + '99',
                    borderColor: splitColors[s],
                    borderWidth: 1.5,
                    borderRadius: 4,
                };
            });

        if (window._dsSplitChart) window._dsSplitChart.destroy();
        window._dsSplitChart = new Chart(splitCanvas.getContext('2d'), {
            type: 'bar',
            data: { labels: labelNames2, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#9ca3af', font: { size: 12 }, padding: 16 }
                    },
                    tooltip: {
                        callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` }
                    }
                },
                scales: {
                    x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: {
                        ticks: { color: '#9ca3af', callback: v => v + '%' },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        title: { display: true, text: '% din split', color: '#9ca3af', font: { size: 11 } }
                    }
                }
            }
        });
    }

    // Sample cards
    const samplesEl = document.getElementById('ds-samples');
    const allSamples = (data.splits.train?.samples || []);
    if (allSamples.length === 0) {
        samplesEl.innerHTML = '<p style="color:var(--text-muted)">No samples available.</p>';
        return;
    }
    samplesEl.innerHTML = '';
    const labelNames = data.label_names;
    allSamples.forEach((s, i) => {
        const color = LABEL_COLORS[labelNames.indexOf(s.label)] || '#9ca3af';
        const card = document.createElement('div');
        card.className = 'ds-sample-card';
        card.innerHTML = `
            <span class="ds-sample-badge" style="background:${color}22;color:${color};border:1px solid ${color}44">
                ${s.label}
            </span>
            <p class="ds-sample-text">"${s.text}${s.text.length >= 220 ? '…' : ''}"</p>
        `;
        samplesEl.appendChild(card);
    });
}

// ─── TRAINING ESTIMATE ───────────────────────────────────────────────────────
// sec/step benchmark (MPS flan-t5-small, measured)
const SEC_PER_STEP = { lora: 0.55, full: 0.85 };

function formatTime(sec) {
    if (sec < 90) return `~${Math.round(sec)}s`;
    if (sec < 3600) return `~${Math.round(sec / 60)}m`;
    return `~${(sec / 3600).toFixed(1)}h`;
}

function updateTrainingEstimate() {
    const samples = parseInt(document.getElementById('train_samples')?.value) || 200;
    const epochsLora = parseInt(document.getElementById('epochs')?.value) || 1;
    const epochsFull = parseInt(document.getElementById('epochs_full')?.value) || 1;
    const batch   = parseInt(document.getElementById('batch_size')?.value) || 8;

    const stepsPerEpoch = Math.ceil(samples / batch);
    const totalStepsLora = stepsPerEpoch * epochsLora;
    const totalStepsFull = stepsPerEpoch * epochsFull;

    document.getElementById('est-steps-epoch').textContent = stepsPerEpoch.toLocaleString();
    document.getElementById('est-steps-total').textContent =
        epochsLora === epochsFull ? totalStepsLora.toLocaleString() : `${totalStepsLora.toLocaleString()} / ${totalStepsFull.toLocaleString()}`;
    document.getElementById('est-time-lora').innerHTML =
        `${formatTime(totalStepsLora * SEC_PER_STEP.lora)} <span style="font-size:0.7rem;color:var(--text-muted)">(LoRA)</span>`;
    document.getElementById('est-time-full').innerHTML =
        `${formatTime(totalStepsFull * SEC_PER_STEP.full)} <span style="font-size:0.7rem;color:var(--text-muted)">(Full FT)</span>`;

    // Warnings
    const warn = document.getElementById('est-warning');
    const warnText = document.getElementById('est-warning-text');
    const messages = [];
    if (epochsLora > 3) messages.push(`${epochsLora} LoRA epochs — risk of overfitting (recommended ≤3).`);
    if (epochsFull > 3) messages.push(`${epochsFull} Full FT epochs — high risk of catastrophic forgetting (recommended ≤3).`);
    if (samples < 100) messages.push(`Only ${samples} examples — too few for generalization.`);
    if (batch > samples / 2) messages.push(`Batch size (${batch}) is very large relative to the dataset (${samples} examples).`);

    if (messages.length > 0) {
        warnText.textContent = messages.join(' ');
        warn.classList.remove('hidden');
    } else {
        warn.classList.add('hidden');
    }
}

// Hook into config inputs
document.addEventListener('DOMContentLoaded', () => {
    ['train_samples', 'epochs', 'epochs_full', 'batch_size', 'learning_rate_lora', 'learning_rate_full'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => {
            updateTrainingEstimate();
            updateLRSchedulerChart();
        });
    });
    updateTrainingEstimate();
    initLRSchedulerChart();
});

// ─── TRUNCATION ANALYSIS ─────────────────────────────────────────────────────
let dsTruncChart = null;

async function loadTruncationStats() {
    try {
        const res = await fetch(`${API_BASE}/api/truncation-stats`);
        if (!res.ok) return;
        const data = await res.json();
        renderTruncationStats(data);
    } catch (e) {
        document.getElementById('ds-trunc-loading').textContent = 'Unavailable (server not running).';
    }
}

function renderTruncationStats(data) {
    document.getElementById('ds-trunc-loading').classList.add('hidden');
    document.getElementById('ds-trunc-content').classList.remove('hidden');
    document.getElementById('ds-max-len').textContent = data.max_input_length;
    document.getElementById('ds-trunc-count').textContent = data.truncated_count.toLocaleString();

    const pctEl = document.getElementById('ds-trunc-pct');
    pctEl.textContent = `${data.truncated_pct}%`;
    pctEl.style.color = data.truncated_pct > 10 ? '#f59e0b' : data.truncated_pct > 0 ? '#10b981' : '#10b981';

    document.getElementById('ds-trunc-avg').textContent = `${data.avg_tokens} tok`;

    // Warning
    const warn = document.getElementById('ds-trunc-warn');
    const warnText = document.getElementById('ds-trunc-warn-text');
    if (data.truncated_pct > 20) {
        warnText.textContent = `${data.truncated_pct}% of examples are truncated — increase MAX_INPUT_LENGTH in config.py (current: ${data.max_input_length}). Max detected: ${data.max_tokens} tokens.`;
        warn.classList.remove('hidden');
    } else if (data.truncated_pct > 5) {
        warnText.textContent = `${data.truncated_pct}% truncation — acceptable, but check the longer examples (max ${data.max_tokens} tok).`;
        warn.classList.remove('hidden');
        warn.style.background = 'rgba(16, 185, 129, 0.08)';
        warn.style.borderColor = 'rgba(16, 185, 129, 0.25)';
        warn.style.color = '#10b981';
    }

    // Histogram chart
    const hist = data.histogram;
    const limitIdx = hist.labels.findIndex(l => {
        const start = parseInt(l.split('–')[0]);
        return start >= data.max_input_length;
    });

    const bgColors = hist.labels.map((_, i) => {
        if (i >= limitIdx && limitIdx !== -1) return 'rgba(255, 69, 91, 0.7)';
        return 'rgba(0, 188, 255, 0.6)';
    });

    const ctx = document.getElementById('ds-trunc-chart').getContext('2d');
    if (dsTruncChart) dsTruncChart.destroy();
    dsTruncChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hist.labels,
            datasets: [{
                label: 'Exemple',
                data: hist.values,
                backgroundColor: bgColors,
                borderWidth: 0,
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: ctx => `Tokeni: ${ctx[0].label}`,
                        label: ctx => ` ${ctx.parsed.y} exemple`,
                        afterLabel: ctx => {
                            const start = parseInt(ctx.label.split('–')[0]);
                            return start >= data.max_input_length ? '⚠ trunchiare' : '';
                        }
                    }
                },
                annotation: limitIdx !== -1 ? {
                    annotations: {
                        limitLine: {
                            type: 'line',
                            xMin: limitIdx - 0.5,
                            xMax: limitIdx - 0.5,
                            borderColor: '#f59e0b',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            label: {
                                display: true,
                                content: `Limit ${data.max_input_length} tok`,
                                color: '#f59e0b',
                                font: { size: 11 },
                                position: 'start'
                            }
                        }
                    }
                } : {}
            },
            scales: {
                x: {
                    ticks: { color: '#9ca3af', maxRotation: 45, font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    ticks: { color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            }
        }
    });
}

// ─── DUPLICATE & LEAKAGE DETECTOR ────────────────────────────────────────────
async function loadDuplicateStats() {
    try {
        const res = await fetch(`${API_BASE}/api/duplicate-stats`);
        if (!res.ok) return;
        const data = await res.json();
        renderDuplicateStats(data);
    } catch (e) {
        const el = document.getElementById('ds-dup-loading');
        if (el) el.textContent = 'Unavailable.';
    }
}

function renderDuplicateStats(data) {
    document.getElementById('ds-dup-loading').classList.add('hidden');
    document.getElementById('ds-dup-content').classList.remove('hidden');

    const dupPct = data.dup_in_train_pct;
    const leakVal = data.leak_train_val;
    const leakTest = data.leak_train_test;

    // Train duplicates
    const dupEl = document.getElementById('ds-dup-train');
    dupEl.textContent = data.dup_in_train === 0
        ? '0 ✓'
        : `${data.dup_in_train} (${dupPct}%)`;
    dupEl.style.color = data.dup_in_train > 0 ? '#f59e0b' : '#10b981';

    // Leak val
    const lvEl = document.getElementById('ds-leak-val');
    lvEl.textContent = leakVal === 0
        ? '0 ✓'
        : `${leakVal} (${data.leak_train_val_pct}%)`;
    lvEl.style.color = leakVal > 0 ? '#ff455b' : '#10b981';

    // Leak test
    const ltEl = document.getElementById('ds-leak-test');
    ltEl.textContent = leakTest === 0
        ? '0 ✓'
        : `${leakTest} (${data.leak_train_test_pct}%)`;
    ltEl.style.color = leakTest > 0 ? '#ff455b' : '#10b981';

    // Warning or OK banner
    const issues = [];
    if (data.dup_in_train > 0)
        issues.push(`${data.dup_in_train} duplicates in train (${dupPct}%) — training will overfit on these examples.`);
    if (leakVal > 0)
        issues.push(`${leakVal} examples from train also appear in validation — evaluation accuracy is artificially inflated.`);
    if (leakTest > 0)
        issues.push(`${leakTest} examples from train also appear in test — final metrics are not trustworthy.`);

    if (issues.length > 0) {
        document.getElementById('ds-dup-warn-text').textContent = issues.join(' ');
        document.getElementById('ds-dup-warn').classList.remove('hidden');
    } else {
        document.getElementById('ds-dup-ok').classList.remove('hidden');
    }
}

