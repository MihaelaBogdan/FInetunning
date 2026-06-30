// ─── STATE MANAGEMENT ───
let liveLossChart = null;
let timeMemChart = null;
let paramsSizeChart = null;
let eventSource = null;

// Culori tematice pentru Chart.js
const colors = {
    lora: '#00bcff',
    loraGlow: 'rgba(0, 188, 255, 0.15)',
    full: '#ff455b',
    fullGlow: 'rgba(255, 69, 91, 0.15)',
    grid: 'rgba(255, 255, 255, 0.05)',
    text: '#9ca3af'
};

// ─── INITIALIZARE LA LOAD ───
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadConfig();
    initLiveLossChart();
    checkTrainingStatus();
    loadAnalytics();
    setupSSE();
    
    // Formular configurare
    document.getElementById('config-form').addEventListener('submit', saveConfig);
    
    // Butoane antrenament
    document.getElementById('start-lora-btn').addEventListener('click', () => startTraining('lora'));
    document.getElementById('start-full-btn').addEventListener('click', () => startTraining('full'));
    
    // Consola
    document.getElementById('clear-console-btn').addEventListener('click', () => {
        document.getElementById('console-output').innerHTML = '';
    });
    
    // Playground
    document.getElementById('predict-btn').addEventListener('click', runPlaygroundInference);
});

// ─── LOGICĂ TABS ───
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
            
            // Re-randare grafice dacă este cazul pentru a evita bug-uri de dimensiune
            if (targetTab === 'analytics') {
                loadAnalytics();
            }
        });
    });
}

// ─── CONFIGURATION ENDPOINTS ───
async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const config = await res.json();
        
        document.getElementById('train_samples').value = config.train_samples;
        document.getElementById('epochs').value = config.epochs;
        document.getElementById('batch_size').value = config.batch_size;
        document.getElementById('learning_rate_lora').value = config.learning_rate_lora;
        document.getElementById('learning_rate_full').value = config.learning_rate_full;
        document.getElementById('lora_r').value = config.lora_r;
        document.getElementById('lora_alpha').value = config.lora_alpha;
        
        writeToConsole("[Sistem] Configurația curentă a fost încărcată cu succes.");
    } catch (err) {
        writeToConsole("[Eroare] Eșec la încărcarea configurației: " + err.message, "error");
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
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (res.ok) {
            writeToConsole("[Sistem] Configurația a fost salvată pe server.");
            alert("Configurația a fost salvată!");
        } else {
            writeToConsole("[Eroare] Serverul a respins configurația: " + data.detail, "error");
        }
    } catch (err) {
        writeToConsole("[Eroare] Eșec la salvarea configurației: " + err.message, "error");
    }
}

// ─── TRAINING CONTROL ───
async function startTraining(method) {
    try {
        const res = await fetch(`/api/train/${method}`, { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            writeToConsole(`[Antrenament] Pornire sesiune pentru: ${method.toUpperCase()}...`);
            setTrainingUI(true, method);
            resetLiveChartForMethod(method);
        } else {
            writeToConsole(`[Eroare] Nu s-a putut porni antrenamentul: ${data.detail}`, "error");
        }
    } catch (err) {
        writeToConsole(`[Eroare] Conexiune eșuată: ${err.message}`, "error");
    }
}

async function checkTrainingStatus() {
    try {
        const res = await fetch('/api/status');
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
        console.error("Eroare la verificarea statusului:", err);
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
        document.getElementById('progress-title').innerText = `Antrenament ${methodTitle} în curs...`;
        
        statusText.innerText = `Antrenare ${method === 'lora' ? 'LoRA' : 'Full FT'}...`;
        statusDot.className = `status-dot pulsing-${method === 'lora' ? 'blue' : 'red'}`;
    } else {
        loraBtn.disabled = false;
        fullBtn.disabled = false;
        saveBtn.disabled = false;
        progressSec.classList.add('hidden');
        
        statusText.innerText = "Sistem Pregătit";
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
    
    eventSource = new EventSource('/api/stream-progress');
    
    eventSource.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'status') {
            if (msg.status === 'started') {
                writeToConsole(`[Server] Antrenamentul ${msg.method.toUpperCase()} a început oficial.`, msg.method + "-color");
                setTrainingUI(true, msg.method);
            } else if (msg.status === 'completed') {
                writeToConsole(`[Server] Antrenamentul ${msg.method.toUpperCase()} s-a finalizat cu succes!`, "system");
                setTrainingUI(false);
                loadAnalytics(); // Reîncarcă analizele cu noile date
            } else if (msg.status === 'failed') {
                writeToConsole(`[Eroare Server] Antrenamentul ${msg.method.toUpperCase()} a eșuat: ${msg.error}`, "error");
                setTrainingUI(false);
            }
        } 
        
        else if (msg.type === 'progress') {
            updateProgressUI(msg.method, msg.data);
            
            // Adăugăm în consolă log-uri de progres
            const stepStr = `[Pas ${msg.data.step}/${msg.data.max_steps}]`;
            const lossStr = msg.data.loss !== null ? `Loss: ${msg.data.loss.toFixed(4)}` : '';
            const evalLossStr = msg.data.eval_loss !== null ? `Eval Loss: ${msg.data.eval_loss.toFixed(4)}` : '';
            
            if (lossStr || evalLossStr) {
                writeToConsole(`${stepStr} ${lossStr} ${evalLossStr}`, msg.method + "-color");
            }
            
            // Adăugăm puncte în graficul de live loss
            if (msg.data.loss !== null) {
                addLiveLossPoint(msg.method, msg.data.step, msg.data.loss);
            }
        }
    };
    
    eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        writeToConsole("[Sistem] Conexiunea la fluxul de progres s-a pierdut. Încercare de reconectare...", "error");
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
                    title: { display: true, text: 'Pași Antrenament', color: colors.text },
                    grid: { color: colors.grid },
                    ticks: { color: colors.text }
                },
                y: {
                    title: { display: true, text: 'Valoare Loss', color: colors.text },
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

function resetLiveChartForMethod(method) {
    const datasetIndex = method === 'lora' ? 0 : 1;
    liveLossChart.data.datasets[datasetIndex].data = [];
    liveLossChart.update();
}

function addLiveLossPoint(method, step, loss) {
    const datasetIndex = method === 'lora' ? 0 : 1;
    liveLossChart.data.datasets[datasetIndex].data.push({ x: step, y: loss });
    
    // Sortăm după pași pentru siguranță
    liveLossChart.data.datasets[datasetIndex].data.sort((a, b) => a.x - b.x);
    
    liveLossChart.update('none'); // Update silențios fără animație pentru viteză
}

// ─── TAB 2: ANALYTICS & GREEN AI ───
async function loadAnalytics() {
    try {
        const res = await fetch('/api/metrics');
        const metrics = await res.json();
        
        const hasLora = metrics.lora !== null;
        const hasFull = metrics.full !== null;
        
        const alertSec = document.getElementById('metrics-alert');
        const contentSec = document.getElementById('analytics-content');
        
        // Controlăm vizibilitatea overlays în tabul de playground
        document.getElementById('overlay-lora').style.display = hasLora ? 'none' : 'flex';
        document.getElementById('overlay-full').style.display = hasFull ? 'none' : 'flex';
        
        if (!hasLora || !hasFull) {
            alertSec.style.display = 'flex';
            contentSec.style.display = 'none';
            
            // Dacă avem doar una dintre ele, populăm parțial KPIs dacă dorim, dar graficele au nevoie de ambele pentru studiu comparativ complet.
            return;
        }
        
        alertSec.style.display = 'none';
        contentSec.style.display = 'flex';
        
        // Populate KPIs
        document.getElementById('kpi-lora-acc').innerText = `${(metrics.lora.final_accuracy * 100).toFixed(2)}%`;
        document.getElementById('kpi-full-acc').innerText = `${(metrics.full.final_accuracy * 100).toFixed(2)}%`;
        
        const energySavings = ((metrics.full.estimated_energy_kwh - metrics.lora.estimated_energy_kwh) / metrics.full.estimated_energy_kwh * 100).toFixed(1);
        document.getElementById('kpi-energy-savings').innerText = `-${energySavings}%`;
        
        // Render Grafice Comparativ
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
        console.error("Eroare la încărcarea analizelor:", err);
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
            labels: ['Timp Antrenare (sec)', 'Peak RAM (MB)'],
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
                y: { grid: { color: colors.grid }, ticks: { color: colors.text }, type: 'logarithmic' } // Scala logaritmică ajută la vizualizarea diferențelor mari
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
            labels: ['Parametri Antrenați (Milioane)', 'Dimensiune Model/Checkpoint (MB)'],
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
        alert("Te rog introdu un text pentru analiză!");
        return;
    }

    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Se procesează...';

    // Resetăm UI-ul predicțiilor și arătăm starea de încărcare
    const models = ['base', 'lora', 'full'];
    models.forEach(m => {
        document.getElementById(`res-${m}-placeholder`).style.display = 'none';
        document.getElementById(`res-${m}-content`).classList.remove('hidden');
        document.getElementById(`res-${m}-label`).innerText = "Analiză...";
        document.getElementById(`res-${m}-val`).innerText = "...";
        document.getElementById(`res-${m}-bar`).style.width = '0%';
    });

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        
        models.forEach(m => {
            const mData = data[m];
            const contentDiv = document.getElementById(`res-${m}-content`);
            const labelDiv = document.getElementById(`res-${m}-label`);
            const valSpan = document.getElementById(`res-${m}-val`);
            const barDiv = document.getElementById(`res-${m}-bar`);
            const iconDiv = document.getElementById(`res-${m}-icon`);
            
            if (mData && mData.available) {
                // Setează clasa pentru etichetă (pozitiv/negativ)
                const labelUpper = mData.label.toUpperCase();
                labelDiv.innerText = labelUpper;
                labelDiv.className = `sentiment-label ${labelUpper.toLowerCase()}`;
                
                // Setează valoarea și bara de încredere
                const confPct = Math.round(mData.confidence * 100);
                valSpan.innerText = `${confPct}%`;
                barDiv.style.width = `${confPct}%`;
                
                // Schimbă iconița în funcție de sentiment
                if (labelUpper === 'POSITIVE') {
                    iconDiv.innerHTML = '<i class="fa-solid fa-face-smile text-green"></i>';
                } else {
                    iconDiv.innerHTML = '<i class="fa-solid fa-face-frown text-red"></i>';
                }
            } else {
                // Dacă modelul nu este disponibil, ascunde conținutul și arată overlay-ul corespunzător
                contentDiv.classList.add('hidden');
                document.getElementById(`res-${m}-placeholder`).style.display = 'block';
                document.getElementById(`res-${m}-placeholder`).innerText = "Model indisponibil";
            }
        });
        
    } catch (err) {
        console.error("Eroare la rularea inferenței:", err);
        alert("A apărut o eroare la comunicarea cu serverul.");
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Analizează Sentimentul';
    }
}
