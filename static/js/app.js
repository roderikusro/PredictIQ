/**
 * =============================================================
 *  Projek Akhir BPS - Main Application JavaScript
 *  Sistem Statistik Prediktif Berbasis Machine Learning
 * =============================================================
 */

// ---- Konfigurasi API ----
const API = {
  dashboard: '/api/dashboard',
  dataset: '/api/dataset',
  modelInfo: '/api/model-info',
  modelComparison: '/api/model-comparison',
  trend: '/api/trend',
  predict: '/api/predict',
  predictCompare: '/api/predict-compare',
  history: '/api/history',
  upload: '/api/upload-csv',
  export: '/api/export',
  exportEda: '/api/export-eda',
  clearHistory: '/api/clear-history',
  eda: '/api/eda',
  infographic: '/api/infographic-data'
};

// ---- State Aplikasi ----
let chartGDP = null;
let chartIndicators = null;
let chartActualVsPred = null;
let chartCoefficients = null;
let chartCompMetrics = null;
let chartFeatureImp = null;
let chartCompAVP = null;
let chartEdaDist = null;
let chartEdaScatter = null;
let edaDataCache = null;

const MODEL_LABELS = {
  linear_regression: 'Linear Regression',
  random_forest: 'Random Forest',
  xgboost: 'XGBoost'
};

function metricValue(value, fallback = 'N/A') {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? fallback : value;
}

function getXgbMetrics(metrics) {
  const xgb = metrics?.xgb || metrics?.xgboost || {};
  return xgb && xgb.available === false ? null : xgb;
}

// ---- Chart.js Global Config ----
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ===========================================================
//                    INISIALISASI APP
// ===========================================================
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initMobileMenu();
  initPredictForm();
  initCompareForm();
  initUpload();
  initExport();
  initClearHistory();
  initDatasetSearch();
  lucide.createIcons();
  loadDashboard();
});

// ===========================================================
//                    NAVIGASI SIDEBAR
// ===========================================================
function initNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const page = link.dataset.page;
      navigateTo(page);
    });
  });
}

function navigateTo(page) {
  // Update nav active
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const activeLink = document.querySelector(`[data-page="${page}"]`);
  if (activeLink) activeLink.classList.add('active');

  // Update page active
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const activePage = document.getElementById(`page-${page}`);
  if (activePage) {
    activePage.classList.add('active');
    activePage.style.animation = 'none';
    activePage.offsetHeight; // trigger reflow
    activePage.style.animation = '';
  }

  // Close mobile sidebar
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebar-overlay').classList.remove('active');

  // Load page data
  switch (page) {
    case 'dashboard': loadDashboard(); break;
    case 'eda': loadEDA(); break;
    case 'prediksi': loadHistory(); break;
    case 'model': loadModelInfo(); break;
    case 'comparison': loadComparison(); break;
    case 'dataset': loadDataset(); break;
    case 'infografis': loadInfographic(); break;
  }
}

// ===========================================================
//                    MOBILE MENU
// ===========================================================
function initMobileMenu() {
  const toggle = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');

  toggle?.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    overlay.classList.toggle('active');
  });
  overlay?.addEventListener('click', () => {
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
  });
}

// ===========================================================
//                    DASHBOARD
// ===========================================================
async function loadDashboard() {
  try {
    const [dashRes, trendRes, histRes] = await Promise.all([
      fetchAPI(API.dashboard),
      fetchAPI(API.trend),
      fetchAPI(API.history)
    ]);

    // Update stat cards
    if (dashRes.status === 'success') {
      const m = dashRes.metrics || {};
      document.getElementById('val-r2').textContent = m.r2_score ?? '—';
      document.getElementById('val-mae').textContent = m.mae ?? '—';
      document.getElementById('val-mse').textContent = m.mse ?? '—';
      document.getElementById('val-total-data').textContent = m.total_data ?? '—';
      document.getElementById('val-train-test').textContent =
        m.train_size && m.test_size ? `Train: ${m.train_size} | Test: ${m.test_size}` : '—';
      const xgbMetrics = getXgbMetrics(m);
      document.getElementById('val-xgb-r2').textContent = metricValue(xgbMetrics?.r2_score);
      document.getElementById('val-xgb-status').textContent = xgbMetrics?.r2_score !== undefined ? 'Aktif' : 'N/A';
      document.getElementById('dashboard-timestamp').textContent = dashRes.timestamp || '';
      
      const bestModelBadge = document.getElementById('dashboard-best-model-badge');
      if (m.model_name) {
          bestModelBadge.style.display = 'flex';
          document.getElementById('dashboard-best-model').textContent = `${m.model_name}`;
      } else {
          bestModelBadge.style.display = 'none';
      }

      // Model status
      const dot = document.getElementById('model-status-dot');
      const text = document.getElementById('model-status-text');
      if (dashRes.model_trained) {
        dot.classList.remove('inactive');
        text.textContent = 'Model Aktif';
        window.globalFeatures = dashRes.features || [];
        window.globalTargetName = dashRes.target_name || 'Target';
        window.globalTimeCols = dashRes.time_cols || {};
        renderDynamicForms();
      } else {
        dot.classList.add('inactive');
        text.textContent = 'Model Tidak Aktif';
      }
    }

    // Render GDP trend chart
    if (trendRes.status === 'success') {
      window.globalTrendData = trendRes.data;
      renderGDPChart(trendRes.data);
      initIndicatorSelectors();
    }

    // Render history
    if (histRes.status === 'success') {
      renderDashboardHistory(histRes.data);
    }

  } catch (err) {
    console.error('Dashboard error:', err);
    showToast('Gagal memuat dashboard', 'error');
  }
}

function renderDynamicForms() {
  if (!window.globalFeatures) return;
  const features = window.globalFeatures;
  const timeCols = window.globalTimeCols || {};
  
  let inputsHtml = '';
  
  // Render time columns first if they exist and are not already in features
  const timeValues = Object.values(timeCols).filter(Boolean);
  const additionalTimeCols = timeValues.filter(t => !features.includes(t));
  
  additionalTimeCols.forEach(t => {
    let label = t.replace(/_/g, ' ') + ' (Waktu)';
    inputsHtml += `
      <div class="form-group" style="background: rgba(139, 92, 246, 0.05); padding: 8px; border-radius: 6px; border-left: 2px solid #8b5cf6;">
        <label style="color: #c4b5fd;">${label}</label>
        <input type="number" name="${t}" value="0" step="any" required>
      </div>
    `;
  });

  // Render ML features
  features.forEach(f => {
    // Generate label cleanly by replacing underscores with spaces
    let label = f.replace(/_/g, ' ');
    if (timeValues.includes(f)) {
        label += ' (Waktu & Fitur)';
    }
    inputsHtml += `
      <div class="form-group">
        <label>${label}</label>
        <input type="number" name="${f}" value="0" step="any" required>
      </div>
    `;
  });

  const predictContainer = document.getElementById('dynamic-prediction-inputs');
  if (predictContainer) {
    predictContainer.innerHTML = inputsHtml;
  }

  const compareContainer = document.getElementById('dynamic-compare-inputs');
  if (compareContainer) {
    compareContainer.innerHTML = inputsHtml;
  }
  
  // Update result label based on target
  document.querySelectorAll('.result-label, .comp-result-unit').forEach(el => {
    if (el.classList.contains('result-label')) {
      el.textContent = `Prediksi ${window.globalTargetName}`;
    } else {
      el.textContent = window.globalTargetName;
    }
  });

  // Ganti teks statis apa pun yang menyebut GDP dengan nama target asli
  document.querySelectorAll('h2, h3, p, span').forEach(el => {
    if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
      if (el.textContent.includes('GDP Growth') || el.textContent.includes('GDP')) {
        el.textContent = el.textContent.replace(/GDP Growth/gi, window.globalTargetName).replace(/GDP/gi, window.globalTargetName);
      }
    }
  });
}

function renderGDPChart(data) {
  const ctx = document.getElementById('chart-gdp-trend');
  if (!ctx) return;
  if (chartGDP) chartGDP.destroy();

  const hasForecast = data.forecast && Array.isArray(data.forecast.target_data) && data.forecast.target_data.length > 0;
  const labels = hasForecast ? (data.combined_labels || data.labels) : data.labels;
  const actualSeries = hasForecast ? (data.actual_series || data.target_data) : data.target_data;
  const forecastSeries = hasForecast ? data.forecast_series : [];

  // Sampling labels agar tidak terlalu padat
  const step = Math.max(1, Math.floor(labels.length / 12));
  const displayLabels = labels.map((l, i) => i % step === 0 ? l : '');

  const datasets = [{
    label: `${data.target_name || 'Target'} Aktual`,
    data: actualSeries,
    borderColor: '#3b82f6',
    backgroundColor: createGradient(ctx, '#3b82f6'),
    borderWidth: 2.5,
    fill: true,
    tension: 0.35,
    pointRadius: 3,
    pointHoverRadius: 6,
    pointBackgroundColor: '#3b82f6',
    pointBorderColor: '#0a0e1a',
    pointBorderWidth: 2,
    spanGaps: false
  }];

  if (hasForecast) {
    datasets.push({
      label: `${data.target_name || 'Target'} Prediksi`,
      data: forecastSeries,
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245, 158, 11, 0.10)',
      borderWidth: 2.5,
      borderDash: [6, 5],
      fill: false,
      tension: 0.35,
      pointRadius: 4,
      pointHoverRadius: 7,
      pointBackgroundColor: '#f59e0b',
      pointBorderColor: '#0a0e1a',
      pointBorderWidth: 2,
      spanGaps: true
    });
  }

  const forecastSummary = document.getElementById('forecast-summary');
  if (forecastSummary) {
    forecastSummary.textContent = hasForecast ? data.forecast.insight : '';
    forecastSummary.style.display = hasForecast ? 'block' : 'none';
  }

  chartGDP = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets
    },
    options: { animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: hasForecast, position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 12 } } },
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.9)',
          borderColor: 'rgba(139,92,246,0.3)',
          borderWidth: 1,
          titleFont: { weight: '600' },
          padding: 12,
          cornerRadius: 8,
        }
      },
      scales: {
        x: {
          ticks: { callback: (val, i) => displayLabels[i] || '', maxRotation: 45, font: { size: 11 } },
          grid: { display: false }
        },
        y: {
          ticks: { callback: v => v + '%', font: { size: 11 } },
          grid: { color: 'rgba(255,255,255,0.04)' }
        }
      }
    }
  });
}

function initIndicatorSelectors() {
  if (!window.globalTrendData || !window.globalTrendData.indicators) return;
  const indicatorsObj = window.globalTrendData.indicators;
  const availableFeatures = Object.keys(indicatorsObj);
  
  if (availableFeatures.length === 0) return;

  // Initialize with the first feature if not already set
  if (!window.selectedIndicators || window.selectedIndicators.length === 0) {
    window.selectedIndicators = [availableFeatures[0]];
  }

  const container = document.getElementById('indicator-selectors');
  if (!container) return;

  // Remove existing dropdowns
  const existingSelects = container.querySelectorAll('.indicator-select-wrapper');
  existingSelects.forEach(el => el.remove());

  const addBtn = document.getElementById('btn-add-indicator');

  // Render current dropdowns
  window.selectedIndicators.forEach((selectedVal, index) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'indicator-select-wrapper';
    wrapper.style.display = 'flex';
    wrapper.style.alignItems = 'center';
    wrapper.style.gap = '4px';

    const select = document.createElement('select');
    select.className = 'form-control';
    select.style.padding = '4px 8px';
    select.style.height = '32px';
    select.style.fontSize = '0.9rem';
    select.style.width = 'auto';
    select.style.minWidth = '120px';
    select.style.background = 'var(--bg-card)';
    select.style.border = '1px solid rgba(255,255,255,0.1)';
    select.style.color = 'var(--text-primary)';
    select.style.borderRadius = '6px';
    select.style.outline = 'none';

    availableFeatures.forEach(f => {
      const option = document.createElement('option');
      option.value = f;
      option.textContent = f;
      if (f === selectedVal) option.selected = true;
      select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
      window.selectedIndicators[index] = e.target.value;
      renderIndicatorsChart();
    });

    wrapper.appendChild(select);

    // Remove button if more than 1 dropdown
    if (window.selectedIndicators.length > 1) {
      const removeBtn = document.createElement('button');
      removeBtn.className = 'btn-icon';
      removeBtn.innerHTML = '&times;';
      removeBtn.style.color = '#ef4444';
      removeBtn.style.padding = '0 6px';
      removeBtn.style.fontSize = '1.2rem';
      removeBtn.title = 'Hapus';
      removeBtn.addEventListener('click', () => {
        window.selectedIndicators.splice(index, 1);
        initIndicatorSelectors();
      });
      wrapper.appendChild(removeBtn);
    }

    container.insertBefore(wrapper, addBtn);
  });

  // Handle add button visibility & click
  addBtn.onclick = () => {
    if (window.selectedIndicators.length < availableFeatures.length) {
      // Find a feature not currently selected
      const unselected = availableFeatures.find(f => !window.selectedIndicators.includes(f)) || availableFeatures[0];
      window.selectedIndicators.push(unselected);
      initIndicatorSelectors();
    }
  };
  addBtn.style.display = window.selectedIndicators.length < availableFeatures.length ? 'block' : 'none';

  renderIndicatorsChart();
}

function renderIndicatorsChart() {
  const data = window.globalTrendData;
  const ctx = document.getElementById('chart-indicators');
  if (!ctx || !data) return;
  if (chartIndicators) chartIndicators.destroy();

  const colors = ['#f59e0b', '#ef4444', '#14b8a6', '#3b82f6', '#ec4899', '#8b5cf6', '#10b981'];
  let datasets = [];
  
  // Create mapping to avoid overlapping labels
  const step = Math.max(1, Math.floor(data.labels.length / 12));
  const displayLabels = data.labels.map((l, i) => i % step === 0 ? l : '');
  
  if (data.indicators && window.selectedIndicators) {
    window.selectedIndicators.forEach((key, i) => {
      if (data.indicators[key]) {
        datasets.push({
          label: key,
          data: data.indicators[key],
          borderColor: colors[i % colors.length],
          borderWidth: 2,
          tension: 0.4,
          pointRadius: 0
        });
      }
    });
  }

  chartIndicators = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: datasets
    },
    options: { animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 11 } } },
        tooltip: { backgroundColor: 'rgba(15,23,42,0.9)', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, cornerRadius: 8 }
      },
      scales: {
        x: { ticks: { display: false }, grid: { display: false } },
        y: { ticks: { callback: v => v + '%', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

function renderDashboardHistory(history) {
  const empty = document.getElementById('empty-history-dashboard');
  const table = document.getElementById('history-table-dashboard');
  const tbody = document.getElementById('history-body-dashboard');
  const badge = document.getElementById('badge-history-count');

  badge.textContent = history.length;

  if (history.length === 0) {
    empty.style.display = 'block';
    table.style.display = 'none';
    return;
  }

  empty.style.display = 'none';
  table.style.display = 'block';

  // Show last 5
  const recent = history.slice(-5).reverse();
  tbody.innerHTML = recent.map((h, i) => `
    <tr>
      <td>${history.length - i}</td>
      <td>${h.timestamp}</td>
      <td><strong style="color:#8b5cf6">${h.prediction}%</strong></td>
      <td>${h.model}</td>
    </tr>
  `).join('');
}

// ===========================================================
//                    PREDIKSI
// ===========================================================
function initPredictForm() {
  const form = document.getElementById('predict-form');
  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    await doPrediction();
  });
}

async function doPrediction() {
  const form = document.getElementById('predict-form');
  const formData = new FormData(form);
  const input = {};
  formData.forEach((val, key) => {
    // model_type is a string, not a number
    input[key] = key === 'model_type' ? val : parseFloat(val);
  });

  showLoading(true);

  try {
    const res = await fetchAPI(API.predict, 'POST', input);

    if (res.status === 'success') {
      const data = res.data;
      // Show result card
      const resultCard = document.getElementById('result-card');
      resultCard.style.display = 'block';
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      // Animate value
      const valEl = document.getElementById('result-value');
      valEl.style.animation = 'none';
      valEl.offsetHeight;
      valEl.style.animation = '';
      valEl.textContent = data.prediction;

      document.getElementById('result-model').textContent = data.model;
      document.getElementById('result-time').textContent = data.timestamp;

      // Update indicator marker position (map -6 to 8 range)
      const minGDP = -6, maxGDP = 8;
      const pct = Math.min(100, Math.max(0, ((data.prediction - minGDP) / (maxGDP - minGDP)) * 100));
      document.getElementById('indicator-marker').style.left = pct + '%';

      // Auto-Insight Display
      const insightContainer = document.getElementById('prediction-insight');
      const insightText = document.getElementById('insight-text');
      if (data.insight && insightContainer && insightText) {
        insightText.textContent = data.insight;
        insightContainer.style.display = 'block';
        // Animasi fade in
        insightContainer.style.opacity = '0';
        setTimeout(() => {
          insightContainer.style.transition = 'opacity 0.5s ease';
          insightContainer.style.opacity = '1';
        }, 50);
      } else if (insightContainer) {
        insightContainer.style.display = 'none';
      }

      showToast(`Prediksi berhasil: ${data.prediction}%`, 'success');
      loadHistory();
    } else {
      showToast(res.message || 'Gagal melakukan prediksi', 'error');
    }
  } catch (err) {
    showToast('Error: ' + err.message, 'error');
  } finally {
    showLoading(false);
  }
}

async function loadHistory() {
  try {
    const res = await fetchAPI(API.history);
    if (res.status === 'success') {
      renderPredictHistory(res.data);
      renderDashboardHistory(res.data);
    }
  } catch (err) { console.error('History error:', err); }
}

function renderPredictHistory(history) {
  const empty = document.getElementById('empty-history-predict');
  const table = document.getElementById('history-table-predict');
  const tbody = document.getElementById('history-body-predict');

  if (history.length === 0) {
    empty.style.display = 'block';
    table.style.display = 'none';
    return;
  }

  empty.style.display = 'none';
  table.style.display = 'block';

  const reversed = [...history].reverse();
  tbody.innerHTML = reversed.map((h, i) => {
    // Collect first 2 features to show
    let featuresList = [];
    if (h.input) {
      const keys = Object.keys(h.input);
      if (keys.length > 0) featuresList.push(`${keys[0]}: ${h.input[keys[0]]}`);
      if (keys.length > 1) featuresList.push(`${keys[1]}: ${h.input[keys[1]]}`);
    }
    
    return `
    <tr>
      <td>${history.length - i}</td>
      <td>${h.timestamp}</td>
      <td colspan="3"><span style="font-size:0.85rem;color:var(--text-secondary);">${featuresList.join(' | ')} ...</span></td>
      <td><strong style="color:#8b5cf6">${h.prediction}</strong></td>
    </tr>
  `}).join('');
}

// ===========================================================
//                    TENTANG MODEL
// ===========================================================
async function loadModelInfo() {
  try {
    const modelType = document.getElementById('model-type-selector')?.value || 'linear_regression';
    const res = await fetchAPI(API.modelInfo + '?type=' + modelType);
    if (res.status !== 'success') return;
    const data = res.data;
    const unavailable = data.available === false || data.error;

    // Overview cards
    document.getElementById('info-model-type').textContent = data.model_type || MODEL_LABELS[modelType];
    document.getElementById('info-library').textContent = data.library || 'N/A';
    document.getElementById('info-scaler').textContent = data.scaler || 'N/A';
    document.getElementById('info-split').textContent = data.train_split && data.test_split ? `${data.train_split} / ${data.test_split}` : 'N/A';

    // Metrics bars
    const mae = data.metrics?.mae;
    const mse = data.metrics?.mse;
    const r2 = data.metrics?.r2_score;

    document.getElementById('metric-mae-val').textContent = metricValue(mae);
    document.getElementById('metric-mse-val').textContent = metricValue(mse);
    document.getElementById('metric-r2-val').textContent = metricValue(r2);

    // Bar fills (normalize for visual)
    setTimeout(() => {
      document.getElementById('bar-mae').style.width = mae !== undefined ? Math.min(100, mae * 20) + '%' : '0%';
      document.getElementById('bar-mse').style.width = mse !== undefined ? Math.min(100, mse * 10) + '%' : '0%';
      document.getElementById('bar-r2').style.width = r2 !== undefined ? Math.max(0, r2 * 100) + '%' : '0%';
    }, 200);

    // Intercept (only for LR)
    const interceptInfo = document.getElementById('intercept-info');
    if (data.intercept !== undefined) {
      document.getElementById('intercept-value').textContent = data.intercept;
      interceptInfo.style.display = 'block';
    } else {
      interceptInfo.style.display = 'none';
    }

    // Actual vs Predicted chart
    if (data.actual_vs_predicted) renderActualVsPredChart(data.actual_vs_predicted);
    else if (chartActualVsPred) {
      chartActualVsPred.destroy();
      chartActualVsPred = null;
    }

    // Toggle description content and chart title
    const descLr = document.getElementById('desc-lr');
    const descRf = document.getElementById('desc-rf');
    const descXgb = document.getElementById('desc-xgb');
    const chartTitle = document.getElementById('coef-chart-title');

    if (modelType === 'random_forest') {
      if (descLr) descLr.style.display = 'none';
      if (descRf) descRf.style.display = 'block';
      if (descXgb) descXgb.style.display = 'none';
      if (chartTitle) chartTitle.textContent = 'Feature Importance (Tingkat Kepentingan)';
    } else if (modelType === 'xgboost') {
      if (descLr) descLr.style.display = 'none';
      if (descRf) descRf.style.display = 'none';
      if (descXgb) descXgb.style.display = 'block';
      if (chartTitle) chartTitle.textContent = unavailable ? 'XGBoost belum tersedia' : 'Feature Importance XGBoost';
    } else {
      if (descLr) descLr.style.display = 'block';
      if (descRf) descRf.style.display = 'none';
      if (descXgb) descXgb.style.display = 'none';
      if (chartTitle) chartTitle.textContent = 'Koefisien Model (Bobot Fitur)';
    }

    // Coefficients / Feature Importance chart
    if (data.coefficients) {
      renderCoefficientsChart(data.coefficients);
    } else if (data.feature_importance) {
      renderCoefficientsChart(data.feature_importance);
    } else if (chartCoefficients) {
      chartCoefficients.destroy();
      chartCoefficients = null;
    }

    if (unavailable) showToast(data.error || 'XGBoost belum tersedia di server', 'info');

  } catch (err) {
    console.error('Model info error:', err);
    showToast('Gagal memuat info model', 'error');
  }
}

// Model type selector change listener
document.getElementById('model-type-selector')?.addEventListener('change', loadModelInfo);

function renderActualVsPredChart(avp) {
  const ctx = document.getElementById('chart-actual-vs-pred');
  if (!ctx) return;
  if (chartActualVsPred) chartActualVsPred.destroy();

  const labels = avp.actual.map((_, i) => `Test ${i + 1}`);

  chartActualVsPred = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Aktual', data: avp.actual, backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 6 },
        { label: 'Prediksi', data: avp.predicted, backgroundColor: 'rgba(139,92,246,0.7)', borderRadius: 6 }
      ]
    },
    options: { animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 12 } } },
        tooltip: { backgroundColor: 'rgba(15,23,42,0.9)', cornerRadius: 8 }
      },
      scales: {
        x: { grid: { display: false } },
        y: { ticks: { callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

function renderCoefficientsChart(coefficients) {
  const ctx = document.getElementById('chart-coefficients');
  if (!ctx) return;
  if (chartCoefficients) chartCoefficients.destroy();

  const labels = Object.keys(coefficients);
  const values = Object.values(coefficients);
  const colors = values.map(v => v >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)');

  chartCoefficients = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Koefisien', data: values, backgroundColor: colors, borderRadius: 6 }]
    },
    options: { animation: false,
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(15,23,42,0.9)', cornerRadius: 8 }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    }
  });
}

// ===========================================================
//                    DATASET VIEWER
// ===========================================================
async function loadDataset() {
  try {
    const res = await fetchAPI(API.dataset);
    if (res.status !== 'success') return;
    const info = res.data;

    // Info cards
    document.getElementById('ds-rows').textContent = info.shape.rows;
    document.getElementById('ds-cols').textContent = info.shape.cols;
    document.getElementById('ds-target').textContent = info.target;
    document.getElementById('ds-features').textContent = info.feature_names.length;

    // Data table
    renderDataTable(info.columns, info.data);

    // Render Model Config
    renderModelConfig(info);

  } catch (err) {
    console.error('Dataset error:', err);
    showToast('Gagal memuat dataset', 'error');
  }
}

function renderModelConfig(info) {
  const targetSelect = document.getElementById('config-target');
  const featuresContainer = document.getElementById('config-features-container');
  if (!targetSelect || !featuresContainer) return;

  const numCols = info.numeric_columns || info.columns;

  // Populate target
  targetSelect.innerHTML = numCols.map(c => 
    `<option value="${c}" ${c === info.target ? 'selected' : ''}>${c}</option>`
  ).join('');

  // Populate features
  featuresContainer.innerHTML = '';
  numCols.map(c => {
    const isChecked = info.feature_names.includes(c);
    const label = document.createElement('label');
    label.className = `chip-checkbox ${isChecked ? 'active' : ''}`;
    label.innerHTML = `<input type="checkbox" name="config-features" value="${c}" ${isChecked ? 'checked' : ''}> ${c}`;
    
    const checkbox = label.querySelector('input');
    checkbox.addEventListener('change', (e) => {
      if (e.target.checked) {
        label.classList.add('active');
      } else {
        label.classList.remove('active');
      }
    });
    featuresContainer.appendChild(label);
  });

  // Populate time configuration dropdowns
  const timeYear = document.getElementById('config-time-year');
  const timeQuarter = document.getElementById('config-time-quarter');
  const timeMonth = document.getElementById('config-time-month');
  const timeDay = document.getElementById('config-time-day');
  
  const populateTimeOptions = (selectElem, selectedVal) => {
    if (!selectElem) return;
    const defaultOpt = `<option value="">-- Tidak Ada --</option>`;
    const options = info.columns.map(c => `<option value="${c}" ${c === selectedVal ? 'selected' : ''}>${c}</option>`).join('');
    selectElem.innerHTML = defaultOpt + options;
  };

  const tc = info.time_cols || {};
  populateTimeOptions(timeYear, tc.year);
  populateTimeOptions(timeQuarter, tc.quarter);
  populateTimeOptions(timeMonth, tc.month);
  populateTimeOptions(timeDay, tc.day);
}

document.getElementById('config-model-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const target = document.getElementById('config-target').value;
  const featureCheckboxes = document.querySelectorAll('input[name="config-features"]:checked');
  const features = Array.from(featureCheckboxes).map(cb => cb.value);

  const time_cols = {
    year: document.getElementById('config-time-year')?.value || null,
    quarter: document.getElementById('config-time-quarter')?.value || null,
    month: document.getElementById('config-time-month')?.value || null,
    day: document.getElementById('config-time-day')?.value || null
  };

  if (features.length === 0) {
    showToast('Minimal pilih 1 fitur!', 'error');
    return;
  }

  showLoading(true);
  try {
    const res = await fetchAPI('/api/configure-model', 'POST', { target, features, time_cols });
    if (res.status === 'success') {
      showToast('Konfigurasi berhasil dan model dilatih ulang!', 'success');
      // Reload everything
      await loadDashboard();
      await loadDataset();
      if (document.getElementById('page-eda').classList.contains('active')) {
          loadEDA();
      }
    } else {
      showToast(res.message || 'Gagal mengatur konfigurasi', 'error');
    }
  } catch (err) {
    showToast('Error: ' + err.message, 'error');
  } finally {
    showLoading(false);
  }
});

let _dsCurrentSortCol = null;
let _dsCurrentSortAsc = true;
let _dsSearchQuery = '';
let _dsCurrentPage = 1;
const _dsRowsPerPage = 10;

function renderDataTable(columns, data) {
  // Store for search
  window._datasetColumns = columns;
  window._datasetData = data;
  
  _dsCurrentSortCol = null;
  _dsCurrentSortAsc = true;
  _dsSearchQuery = document.getElementById('dataset-search')?.value || '';
  _dsCurrentPage = 1;
  
  updateDatasetTable();
}

function updateDatasetTable() {
  if (!window._datasetData) return;
  
  let data = [...window._datasetData];

  // 1. Filter
  if (_dsSearchQuery) {
    const q = _dsSearchQuery.toLowerCase();
    data = data.filter(row => 
      Object.values(row).some(v => String(v).toLowerCase().includes(q))
    );
  }

  // 2. Sort
  if (_dsCurrentSortCol) {
    data.sort((a, b) => {
      let valA = a[_dsCurrentSortCol];
      let valB = b[_dsCurrentSortCol];
      
      let numA = parseFloat(valA);
      let numB = parseFloat(valB);
      
      if (!isNaN(numA) && !isNaN(numB)) {
        return _dsCurrentSortAsc ? numA - numB : numB - numA;
      }
      
      valA = String(valA).toLowerCase();
      valB = String(valB).toLowerCase();
      if (valA < valB) return _dsCurrentSortAsc ? -1 : 1;
      if (valA > valB) return _dsCurrentSortAsc ? 1 : -1;
      return 0;
    });
  }

  // 3. Pagination calculation
  const totalRows = data.length;
  const totalPages = Math.ceil(totalRows / _dsRowsPerPage) || 1;
  
  if (_dsCurrentPage > totalPages) _dsCurrentPage = totalPages;
  if (_dsCurrentPage < 1) _dsCurrentPage = 1;

  // 4. Slice current page
  const startIdx = (_dsCurrentPage - 1) * _dsRowsPerPage;
  const pageData = data.slice(startIdx, startIdx + _dsRowsPerPage);

  // 5. Update Headers
  const thead = document.getElementById('dataset-thead');
  const cols = window._datasetColumns;
  thead.innerHTML = '<tr>' + cols.map(c => {
    let icon = '';
    if (c === _dsCurrentSortCol) {
      icon = _dsCurrentSortAsc ? ' <span style="font-size: 0.8em">▲</span>' : ' <span style="font-size: 0.8em">▼</span>';
    } else {
      icon = ' <span style="font-size: 0.8em; opacity: 0.3">↕</span>';
    }
    return `<th style="cursor: pointer; user-select: none;" onclick="window.handleSort('${c}')">${c}${icon}</th>`;
  }).join('') + '</tr>';

  // 6. Render Body
  const tbody = document.getElementById('dataset-tbody');
  if (pageData.length === 0) {
    tbody.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;">Data tidak ditemukan</td></tr>`;
  } else {
    tbody.innerHTML = pageData.map(row =>
      '<tr>' + cols.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>'
    ).join('');
  }

  // 7. Render Pagination
  renderPagination(totalPages);
}

function renderPagination(totalPages) {
  const container = document.getElementById('dataset-pagination');
  if (!container) return;

  if (totalPages <= 1) {
    container.innerHTML = '';
    return;
  }

  let html = `<button class="btn btn-sm btn-ghost" onclick="window.handlePageChange(${_dsCurrentPage - 1})" ${_dsCurrentPage === 1 ? 'disabled' : ''}>Prev</button>`;
  
  for (let i = 1; i <= totalPages; i++) {
    if (i === _dsCurrentPage) {
      html += `<button class="btn btn-sm btn-primary" onclick="window.handlePageChange(${i})">${i}</button>`;
    } else {
      html += `<button class="btn btn-sm btn-ghost" onclick="window.handlePageChange(${i})">${i}</button>`;
    }
  }

  html += `<button class="btn btn-sm btn-ghost" onclick="window.handlePageChange(${_dsCurrentPage + 1})" ${_dsCurrentPage === totalPages ? 'disabled' : ''}>Next</button>`;

  container.innerHTML = html;
}

window.handlePageChange = function(newPage) {
  _dsCurrentPage = newPage;
  updateDatasetTable();
};

window.handleSort = function(col) {
  if (_dsCurrentSortCol === col) {
    _dsCurrentSortAsc = !_dsCurrentSortAsc;
  } else {
    _dsCurrentSortCol = col;
    _dsCurrentSortAsc = true;
  }
  updateDatasetTable();
};


function initDatasetSearch() {
  const input = document.getElementById('dataset-search');
  input?.addEventListener('input', () => {
    _dsSearchQuery = input.value;
    _dsCurrentPage = 1;
    updateDatasetTable();
  });
}

// ===========================================================
//                    UPLOAD CSV / EXCEL
// ===========================================================
function initUpload() {
  const fileInput = document.getElementById('file-upload-input');
  fileInput?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const statusCard = document.getElementById('upload-status-card');
    statusCard.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(API.upload, { method: 'POST', body: formData });
      const data = await res.json();

      if (data.status === 'success') {
        showToast('Dataset berhasil diunggah! Model dilatih ulang.', 'success');
        loadDataset();
        loadDashboard();
      } else {
        showToast(data.message || 'Gagal mengunggah', 'error');
      }
    } catch (err) {
      showToast('Error upload: ' + err.message, 'error');
    } finally {
      statusCard.style.display = 'none';
      fileInput.value = '';
    }
  });
}

// ===========================================================
//                    EXPORT & CLEAR
// ===========================================================
function initExport() {
  document.getElementById('btn-export-history-csv')?.addEventListener('click', () => {
    window.location.href = API.export + '?format=csv';
  });
  document.getElementById('btn-export-history-excel')?.addEventListener('click', () => {
    window.location.href = API.export + '?format=excel';
  });
}

function initClearHistory() {
  document.getElementById('btn-clear-history')?.addEventListener('click', async () => {
    if (!confirm('Hapus semua riwayat prediksi?')) return;
    try {
      const res = await fetchAPI(API.clearHistory, 'POST');
      if (res.status === 'success') {
        showToast('Riwayat berhasil dihapus', 'success');
        loadHistory();
      }
    } catch (err) {
      showToast('Gagal menghapus riwayat', 'error');
    }
  });
}

// ===========================================================
//                    UTILITIES
// ===========================================================
async function fetchAPI(url, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  return res.json();
}

function createGradient(ctx, color) {
  const canvas = ctx.getContext ? ctx : ctx.canvas;
  const c = canvas.getContext('2d');
  const g = c.createLinearGradient(0, 0, 0, 300);
  g.addColorStop(0, color + '33');
  g.addColorStop(1, color + '00');
  return g;
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const icons = { success: 'check-circle', error: 'alert-circle', info: 'info' };
  toast.innerHTML = `<i data-lucide="${icons[type] || 'info'}"></i><span>${message}</span>`;
  container.appendChild(toast);
  lucide.createIcons({ nodes: [toast] });
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(100%)'; setTimeout(() => toast.remove(), 300); }, 4000);
}

function showLoading(show) {
  const el = document.getElementById('loading-overlay');
  if (show) el.classList.add('active');
  else el.classList.remove('active');
}

// Refresh buttons & Export
document.getElementById('btn-refresh-trend')?.addEventListener('click', loadDashboard);
document.getElementById('btn-export-eda-csv')?.addEventListener('click', () => { 
  const removeOutliers = document.getElementById('toggle-remove-outliers')?.checked || false;
  window.location.href = API.exportEda + '?format=csv&remove_outliers=' + removeOutliers; 
});
document.getElementById('btn-export-eda-excel')?.addEventListener('click', () => { 
  const removeOutliers = document.getElementById('toggle-remove-outliers')?.checked || false;
  window.location.href = API.exportEda + '?format=excel&remove_outliers=' + removeOutliers; 
});
document.getElementById('toggle-remove-outliers')?.addEventListener('change', loadEDA);

// ===========================================================
//                    PERBANDINGAN MODEL
// ===========================================================
async function loadComparison() {
  try {
    const res = await fetchAPI(API.modelComparison);
    if (res.status !== 'success') return;
    const d = res.data;

    // Best model banner
    document.getElementById('best-model-name').textContent = d.best_model;
    const s = d.summary;
    document.getElementById('best-model-reason').textContent =
      `R2 terbaik: ${s.r2_winner} | MAE terendah: ${s.mae_winner} | RMSE terendah: ${s.rmse_winner}`;

    // LR metrics
    const lr = d.linear_regression.metrics;
    document.getElementById('comp-lr-mae').textContent = lr.mae;
    document.getElementById('comp-lr-rmse').textContent = lr.rmse;
    document.getElementById('comp-lr-r2').textContent = lr.r2_score;

    // RF metrics
    const rf = d.random_forest.metrics;
    document.getElementById('comp-rf-mae').textContent = rf.mae;
    document.getElementById('comp-rf-rmse').textContent = rf.rmse;
    document.getElementById('comp-rf-r2').textContent = rf.r2_score;

    // XGBoost metrics
    const xgbBlock = d.xgboost || d.xgb || {};
    const xgbAvailable = xgbBlock.available !== false && xgbBlock.metrics && Object.keys(xgbBlock.metrics).length > 0;
    const xgb = xgbAvailable ? xgbBlock.metrics : null;
    document.getElementById('comp-xgb-mae').textContent = metricValue(xgb?.mae);
    document.getElementById('comp-xgb-rmse').textContent = metricValue(xgb?.rmse);
    document.getElementById('comp-xgb-r2').textContent = metricValue(xgb?.r2_score);

    // Badges & winner highlight
    const lrCard = document.getElementById('comp-card-lr');
    const rfCard = document.getElementById('comp-card-rf');
    const xgbCard = document.getElementById('comp-card-xgb');
    lrCard.classList.remove('winner');
    rfCard.classList.remove('winner');
    xgbCard.classList.remove('winner');
    ['badge-lr', 'badge-rf', 'badge-xgb'].forEach(id => {
      const badge = document.getElementById(id);
      badge.textContent = '';
      badge.style.background = '';
      badge.style.color = '';
    });

    if (d.best_model === 'XGBoost' && xgbAvailable) {
      xgbCard.classList.add('winner');
      document.getElementById('badge-xgb').textContent = 'Best';
      document.getElementById('badge-xgb').style.background = 'rgba(245,158,11,0.2)';
      document.getElementById('badge-xgb').style.color = '#f59e0b';
    } else if (d.best_model === 'Random Forest') {
      rfCard.classList.add('winner');
      document.getElementById('badge-rf').textContent = 'Best';
      document.getElementById('badge-rf').style.background = 'rgba(245,158,11,0.2)';
      document.getElementById('badge-rf').style.color = '#f59e0b';
    } else {
      lrCard.classList.add('winner');
      document.getElementById('badge-lr').textContent = 'Best';
      document.getElementById('badge-lr').style.background = 'rgba(245,158,11,0.2)';
      document.getElementById('badge-lr').style.color = '#f59e0b';
    }
    if (!xgbAvailable) document.getElementById('badge-xgb').textContent = 'N/A';

    // Charts
    renderCompMetricsChart(lr, rf, xgb);
    const importanceTitle = document.getElementById('feature-importance-title');
    const importanceSource = xgbAvailable && d.xgboost?.feature_importance ? d.xgboost.feature_importance : d.random_forest.feature_importance;
    if (importanceTitle) importanceTitle.textContent = xgbAvailable ? 'Feature Importance (XGBoost)' : 'Feature Importance (Random Forest)';
    renderFeatureImportanceChart(importanceSource || {});
    renderCompAVPChart(
      d.linear_regression.actual_vs_predicted,
      d.random_forest.actual_vs_predicted,
      xgbAvailable ? d.xgboost.actual_vs_predicted : null
    );

    lucide.createIcons();
  } catch (err) {
    console.error('Comparison error:', err);
    showToast('Gagal memuat perbandingan model', 'error');
  }
}

function renderCompMetricsChart(lr, rf, xgb = null) {
  const ctx = document.getElementById('chart-comparison-metrics');
  if (!ctx) return;
  if (chartCompMetrics) chartCompMetrics.destroy();

  const datasets = [
    {
      label: 'Linear Regression',
      data: [lr.mae, lr.rmse, Math.max(0, lr.r2_score)],
      backgroundColor: 'rgba(59,130,246,0.7)',
      borderRadius: 8
    },
    {
      label: 'Random Forest',
      data: [rf.mae, rf.rmse, Math.max(0, rf.r2_score)],
      backgroundColor: 'rgba(20,184,166,0.7)',
      borderRadius: 8
    }
  ];

  if (xgb) {
    datasets.push({
      label: 'XGBoost',
      data: [xgb.mae, xgb.rmse, Math.max(0, xgb.r2_score)],
      backgroundColor: 'rgba(245,158,11,0.75)',
      borderRadius: 8
    });
  }

  chartCompMetrics = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['MAE', 'RMSE', 'R² Score'],
      datasets
    },
    options: { animation: false,
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 12 } } },
        tooltip: { backgroundColor: 'rgba(15,23,42,0.9)', cornerRadius: 8 }
      },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

function renderFeatureImportanceChart(importance) {
  const ctx = document.getElementById('chart-feature-importance');
  if (!ctx) return;
  if (chartFeatureImp) chartFeatureImp.destroy();

  const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);
  const labels = sorted.map(e => e[0]);
  const values = sorted.map(e => e[1]);

  chartFeatureImp = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance',
        data: values,
        backgroundColor: values.map((_, i) => {
          const colors = ['#14b8a6', '#06b6d4', '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#f59e0b', '#ef4444'];
          return colors[i % colors.length];
        }),
        borderRadius: 6
      }]
    },
    options: { animation: false,
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(15,23,42,0.9)', cornerRadius: 8 }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    }
  });
}

function renderCompAVPChart(lrAVP, rfAVP, xgbAVP = null) {
  const ctx = document.getElementById('chart-comparison-avp');
  if (!ctx) return;
  if (chartCompAVP) chartCompAVP.destroy();

  const labels = lrAVP.actual.map((_, i) => `Test ${i + 1}`);
  const datasets = [
    { label: 'Aktual', data: lrAVP.actual, backgroundColor: 'rgba(148,163,184,0.5)', borderRadius: 6 },
    { label: 'LR Prediksi', data: lrAVP.predicted, backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 6 },
    { label: 'RF Prediksi', data: rfAVP.predicted, backgroundColor: 'rgba(20,184,166,0.7)', borderRadius: 6 }
  ];

  if (xgbAVP) {
    datasets.push({ label: 'XGBoost Prediksi', data: xgbAVP.predicted, backgroundColor: 'rgba(245,158,11,0.75)', borderRadius: 6 });
  }

  chartCompAVP = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets
    },
    options: { animation: false,
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 12 } } },
        tooltip: { backgroundColor: 'rgba(15,23,42,0.9)', cornerRadius: 8 }
      },
      scales: {
        x: { grid: { display: false } },
        y: { ticks: { callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

// Compare Form
function initCompareForm() {
  const form = document.getElementById('compare-form');
  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const input = {};
    formData.forEach((val, key) => { input[key] = parseFloat(val); });

    showLoading(true);
    try {
      const res = await fetchAPI(API.predictCompare, 'POST', input);
      if (res.status === 'success') {
        const d = res.data;
        const card = document.getElementById('compare-result-card');
        card.style.display = 'block';
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        document.getElementById('comp-pred-lr').textContent = d.linear_regression.prediction;
        document.getElementById('comp-pred-rf').textContent = d.random_forest.prediction;
        document.getElementById('comp-pred-xgb').textContent = d.xgboost?.prediction ?? 'N/A';
        document.getElementById('comp-best-text').textContent =
          `Model Terbaik: ${d.best_model} (Prediksi: ${d.best_prediction}%)`;

        // Render insights
        const list = document.getElementById('insight-list');
        list.innerHTML = d.insight.map(i => `<li>${i}</li>`).join('');

        lucide.createIcons();
        showToast('Perbandingan prediksi berhasil!', 'success');
      } else {
        showToast(res.message || 'Gagal', 'error');
      }
    } catch (err) {
      showToast('Error: ' + err.message, 'error');
    } finally {
      showLoading(false);
    }
  });
}

// ===========================================================
//                    EDA (Exploratory Data Analysis)
// ===========================================================
async function loadEDA() {
  try {
    showLoading(true);
    const removeOutliers = document.getElementById('toggle-remove-outliers')?.checked || false;
    const res = await fetchAPI(API.eda + '?remove_outliers=' + removeOutliers);
    if (res.status !== 'success') return;
    edaDataCache = res.data;

    // 1. Insights
    const list = document.getElementById('eda-insights-list');
    list.innerHTML = edaDataCache.insights.map(i => `<li>${i}</li>`).join('');

    // 2. Heatmap Selectors & Initial Render
    const heatmapSelectors = document.getElementById('heatmap-var-selectors');
    heatmapSelectors.innerHTML = '';
    const numCols = edaDataCache.numeric_columns || edaDataCache.columns;
    let selectedHeatmapCols = [...numCols];

    numCols.forEach(col => {
      const label = document.createElement('label');
      label.className = 'chip-checkbox active';
      label.innerHTML = `<input type="checkbox" value="${col}" checked> ${col}`;
      
      const checkbox = label.querySelector('input');
      checkbox.addEventListener('change', (e) => {
        if (e.target.checked) {
          label.classList.add('active');
          if (!selectedHeatmapCols.includes(col)) selectedHeatmapCols.push(col);
        } else {
          label.classList.remove('active');
          selectedHeatmapCols = selectedHeatmapCols.filter(c => c !== col);
        }
        // Maintain original column order
        selectedHeatmapCols.sort((a, b) => numCols.indexOf(a) - numCols.indexOf(b));
        renderEdaHeatmap(edaDataCache.correlation, selectedHeatmapCols);
      });
      heatmapSelectors.appendChild(label);
    });

    renderEdaHeatmap(edaDataCache.correlation, selectedHeatmapCols);

    // 3. Descriptive Stats Table
    const tbody = document.querySelector('#eda-desc-table tbody');
    tbody.innerHTML = '';
    for (const col of numCols) {
      const desc = edaDataCache.descriptive[col];
      const out = edaDataCache.outliers[col] || 0;
      const mis = edaDataCache.missing_values[col] || 0;
      
      if (!desc) continue; // Skip if no descriptive stats
      
      tbody.innerHTML += `
        <tr>
          <td><strong>${col}</strong></td>
          <td>${desc.count ? desc.count : '-'}</td>
          <td>${desc.mean ? desc.mean.toFixed(2) : '-'}</td>
          <td>${desc['50%'] ? desc['50%'].toFixed(2) : '-'}</td>
          <td>${desc.std ? desc.std.toFixed(2) : '-'}</td>
          <td>${desc.min ? desc.min.toFixed(2) : '-'}</td>
          <td>${desc.max ? desc.max.toFixed(2) : '-'}</td>
          <td style="color: ${out > 0 ? '#ef4444' : 'inherit'}; font-weight: ${out > 0 ? 'bold' : 'normal'}">${out}</td>
          <td style="color: ${mis > 0 ? '#ef4444' : 'inherit'}; font-weight: ${mis > 0 ? 'bold' : 'normal'}">${mis}</td>
        </tr>
      `;
    }

    // 4. Populating Selects
    const distSelect = document.getElementById('eda-dist-select');
    const xSelect = document.getElementById('eda-scatter-x');
    const ySelect = document.getElementById('eda-scatter-y');
    
    [distSelect, xSelect, ySelect].forEach(sel => sel.innerHTML = '');
    numCols.forEach(c => {
      distSelect.add(new Option(c, c));
      xSelect.add(new Option(c, c));
      ySelect.add(new Option(c, c));
    });

    if (edaDataCache.target && numCols.includes(edaDataCache.target)) {
      distSelect.value = edaDataCache.target;
    }

    // Default scatter selections
    if (numCols.length >= 2) {
      xSelect.value = numCols[0];
      ySelect.value = numCols[1];
    } else if (numCols.length === 1) {
      xSelect.value = numCols[0];
      ySelect.value = numCols[0];
    }

    // Initial Renders
    renderEdaDist(distSelect.value);
    if (numCols.length > 0) {
      renderEdaScatter(xSelect.value, ySelect.value);
    }

    // Event listeners
    distSelect.onchange = (e) => renderEdaDist(e.target.value);
    xSelect.onchange = (e) => renderEdaScatter(xSelect.value, ySelect.value);
    ySelect.onchange = (e) => renderEdaScatter(xSelect.value, ySelect.value);

    lucide.createIcons();
  } catch (err) {
    console.error('EDA error:', err);
    showToast('Gagal memuat data EDA', 'error');
  } finally {
    showLoading(false);
  }
}

function renderEdaHeatmap(corr, columns) {
  const container = document.getElementById('eda-heatmap-container');
  container.innerHTML = '';
  
  if (!columns || columns.length === 0) {
    container.innerHTML = '<div style="grid-column: 1 / -1; padding: 20px; text-align: center; color: var(--text-muted);">Pilih minimal 1 variabel untuk melihat korelasi.</div>';
    container.style.gridTemplateColumns = '1fr';
    return;
  }

  const n = columns.length;
  container.style.gridTemplateColumns = `100px repeat(${n}, 1fr)`;

  // Header row
  container.appendChild(document.createElement('div')); // empty top-left
  columns.forEach(c => {
    const el = document.createElement('div');
    el.className = 'heatmap-label top';
    el.textContent = c;
    container.appendChild(el);
  });

  // Rows
  columns.forEach(rowCol => {
    // Row label
    const label = document.createElement('div');
    label.className = 'heatmap-label';
    label.textContent = rowCol;
    container.appendChild(label);

    columns.forEach(colCol => {
      const val = corr[rowCol][colCol];
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.textContent = val.toFixed(2);
      
      // Color logic (Blue for positive, Red for negative)
      // opacity scaled by absolute value
      const absVal = Math.abs(val);
      if (val >= 0) {
        cell.style.backgroundColor = `rgba(59, 130, 246, ${absVal})`;
      } else {
        cell.style.backgroundColor = `rgba(239, 68, 68, ${absVal})`;
      }
      
      if (absVal < 0.4) cell.style.color = '#cbd5e1'; // light text for dark cells

      cell.title = `${rowCol} vs ${colCol}: ${val.toFixed(3)}`;
      container.appendChild(cell);
    });
  });
}

function renderEdaDist(col) {
  if (!edaDataCache) return;
  const ctx = document.getElementById('chart-eda-dist');
  if (!ctx) return;
  if (chartEdaDist) chartEdaDist.destroy();

  const toNumericValues = values => values
    .map(v => Number(v))
    .filter(v => Number.isFinite(v));

  const actualData = toNumericValues(edaDataCache.scatter_data.map(d => d[col]));
  const forecast = edaDataCache.forecast || window.globalTrendData?.forecast;
  const targetName = edaDataCache.target || forecast?.target_name || window.globalTargetName;
  let predictedData = [];

  if (forecast) {
    if (col === targetName) {
      predictedData = toNumericValues(forecast.target_data || []);
    } else if (forecast.feature_data && forecast.feature_data[col]) {
      predictedData = toNumericValues(forecast.feature_data[col]);
    }
  }

  if (actualData.length === 0) return;

  // Histogram aktual dan prediksi memakai bin yang sama agar perbandingan adil.
  const rangeData = actualData.concat(predictedData);
  let min = Math.min(...rangeData);
  let max = Math.max(...rangeData);
  if (min === max) {
    min -= 1;
    max += 1;
  }

  const bins = 10;
  const step = (max - min) / bins;
  const labels = [];
  
  for(let i=0; i<bins; i++) {
    labels.push(`${(min + i*step).toFixed(1)} - ${(min + (i+1)*step).toFixed(1)}`);
  }

  const buildHist = values => {
    const hist = new Array(bins).fill(0);
    values.forEach(val => {
      let idx = Math.floor((val - min) / step);
      if (idx < 0) idx = 0;
      if (idx >= bins) idx = bins - 1;
      hist[idx]++;
    });
    return hist;
  };

  const datasets = [{
    label: `Aktual ${col}`,
    data: buildHist(actualData),
    backgroundColor: 'rgba(59, 130, 246, 0.72)',
    borderColor: 'rgba(59, 130, 246, 0.95)',
    borderWidth: 1,
    borderRadius: 4
  }];

  if (predictedData.length > 0) {
    datasets.push({
      label: `${col === targetName ? 'Prediksi' : 'Proyeksi'} ${col}`,
      data: buildHist(predictedData),
      backgroundColor: 'rgba(245, 158, 11, 0.72)',
      borderColor: 'rgba(245, 158, 11, 0.95)',
      borderWidth: 1,
      borderRadius: 4
    });
  }

  chartEdaDist = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets
    },
    options: { animation: false,
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: predictedData.length > 0, position: 'bottom', labels: { usePointStyle: true, padding: 14, font: { size: 11 } } },
        tooltip: { backgroundColor: 'rgba(15,23,42,0.9)' }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
      }
    }
  });
}

function renderEdaScatter(xCol, yCol) {
  if (!edaDataCache) return;
  const ctx = document.getElementById('chart-eda-scatter');
  if (!ctx) return;
  if (chartEdaScatter) chartEdaScatter.destroy();

  const scatterData = edaDataCache.scatter_data.map(d => ({
    x: d[xCol],
    y: d[yCol]
  }));

  chartEdaScatter = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: `${yCol} vs ${xCol}`,
        data: scatterData,
        backgroundColor: '#14b8a6',
        pointRadius: 6,
        pointHoverRadius: 8
      }]
    },
    options: { animation: false,
      responsive: true, maintainAspectRatio: false,
      plugins: {
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.9)',
          callbacks: {
            label: (ctx) => `(${ctx.parsed.x}, ${ctx.parsed.y})`
          }
        }
      },
      scales: {
        x: { 
          title: { display: true, text: xCol, color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.04)' } 
        },
        y: { 
          title: { display: true, text: yCol, color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.04)' } 
        }
      }
    }
  });
}

function exportEDATable() {
  const table = document.getElementById('eda-desc-table');
  if (!table) return;
  
  let csvContent = "";
  
  // Get headers
  const headers = Array.from(table.querySelectorAll('thead th')).map(th => `"${th.textContent.trim()}"`);
  csvContent += headers.join(",") + "\n";
  
  // Get rows
  const rows = table.querySelectorAll('tbody tr');
  rows.forEach(tr => {
    const cols = Array.from(tr.querySelectorAll('td')).map(td => `"${td.textContent.trim()}"`);
    csvContent += cols.join(",") + "\n";
  });
  
  // Create and trigger download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const dateStr = new Date().toISOString().split('T')[0];
  a.download = `Statistik_Deskriptif_${dateStr}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ===========================================================
//                    EXPORT TO IMAGE (PNG)
// ===========================================================
async function exportToImage(elementId, fileName) {
  const element = document.getElementById(elementId);
  if (!element) return;
  
  try {
    showToast('Mempersiapkan gambar...', 'info');
    
    // Hide export buttons so they don't show up in the image
    const btns = element.querySelectorAll('[id^="btn-export-"]');
    btns.forEach(btn => btn.style.display = 'none');
    
    // Add a slight padding to the element for a better look
    const originalPadding = element.style.padding;
    element.style.padding = '20px';
    
    // Disable animations temporarily to prevent html2canvas opacity bugs
    const activeLayout = element.querySelector('.infog-layout.active');
    const originalAnimation = activeLayout ? activeLayout.style.animation : '';
    if (activeLayout) {
      activeLayout.style.animation = 'none';
      activeLayout.style.opacity = '1';
    }
    
    const originalScrollY = window.scrollY;
    const originalScrollX = window.scrollX;
    window.scrollTo(0, 0);
    
    // Wait for any pending Chart.js animations to complete before capturing
    await new Promise(r => setTimeout(r, 800));
    
    const canvas = await html2canvas(element, {
      backgroundColor: '#0a0e1a', // Match app background
      scale: 2, // High resolution
      logging: false,
      useCORS: true
    });
    
    window.scrollTo(originalScrollX, originalScrollY);
    
    // Restore styling
    element.style.padding = originalPadding;
    if (activeLayout) {
      activeLayout.style.animation = originalAnimation;
      activeLayout.style.opacity = '';
    }
    btns.forEach(btn => btn.style.display = '');
    
    const image = canvas.toDataURL("image/png");
    const a = document.createElement('a');
    a.href = image;
    const dateStr = new Date().toISOString().split('T')[0];
    a.download = `${fileName}_${dateStr}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    showToast('Gambar berhasil diunduh!', 'success');
  } catch (err) {
    console.error('Error exporting image:', err);
    showToast('Gagal mengekspor gambar.', 'error');
  }
}

// ===========================================================
//                    INFOGRAFIS OTOMATIS
// ===========================================================
let infogChartTrend = null;
let infogChartImportance = null;
let infogChartHistogram = null;
let infogChartScatter = null;
let infogCurrentLayout = 1;
let infogCurrentTheme = 'dark';
let infogBgColor = '#0d1225';
let infogShapeColor = '#1e293b';
let infogChartColor = '#8b5cf6';
let infogTextColor = '#f8fafc';
let infogCachedData = null;

// --- Toolbar Initialization ---
function initInfographicToolbar() {
  const canvas = document.getElementById('infographic-canvas');
  
  const bgInput = document.getElementById('infog-color-bg');
  const shapeInput = document.getElementById('infog-color-shape');
  const chartInput = document.getElementById('infog-color-chart');
  const textInput = document.getElementById('infog-color-text');

  function applyColors() {
    infogBgColor = bgInput.value;
    infogShapeColor = shapeInput.value;
    infogChartColor = chartInput.value;
    infogTextColor = textInput.value;

    // Use a <style> block for html2canvas compatibility. 
    // html2canvas notoriously fails to resolve CSS variables defined via inline styles.
    let styleTag = document.getElementById('infog-html2canvas-fix');
    if (!styleTag) {
      styleTag = document.createElement('style');
      styleTag.id = 'infog-html2canvas-fix';
      document.head.appendChild(styleTag);
    }
    
    styleTag.innerHTML = `
      #infographic-canvas {
        --infog-bg: ${infogBgColor};
        --infog-shape: ${infogShapeColor};
        --infog-chart: ${infogChartColor};
        --infog-text: ${infogTextColor};
        --text-primary: ${infogTextColor} !important;
        --text-secondary: ${infogTextColor} !important;
        --text-muted: ${infogTextColor} !important;
      }
      #infographic-canvas h1,
      #infographic-canvas h2,
      #infographic-canvas p,
      #infographic-canvas span,
      #infographic-canvas .infog-stat-value,
      #infographic-canvas .infog-stat-label,
      #infographic-canvas .infog-metric-ring-val,
      #infographic-canvas .infog-compare-val,
      #infographic-canvas .infog-compare-name,
      #infographic-canvas .infog-corr-name,
      #infographic-canvas .infog-narrative-item,
      #infographic-canvas .infog-insight-card,
      #infographic-canvas .infog-footer,
      #infographic-canvas .infog-meta {
        color: ${infogTextColor} !important;
      }
      #infographic-canvas .infog-header h1 {
        -webkit-text-fill-color: ${infogTextColor} !important;
        background-image: none !important;
        background: none !important;
      }
      #infographic-canvas .infog-stat-icon i,
      #infographic-canvas .infog-corr-val.positive,
      #infographic-canvas .infog-corr-val.negative,
      #infographic-canvas .infog-model-name,
      #infographic-canvas .infog-metric-big,
      #infographic-canvas .infog-insight-icon,
      #infographic-canvas .infog-logo-icon i,
      #infographic-canvas .infog-narrative-num {
        color: inherit !important;
      }
    `;

    canvas.style.setProperty('--infog-bg', infogBgColor);
    canvas.style.setProperty('--infog-shape', infogShapeColor);
    canvas.style.setProperty('--infog-chart', infogChartColor);
    canvas.style.setProperty('--infog-text', infogTextColor);

    // Apply specific chart color overrides manually (R2 ring, text, icons)
    applyInfographicAccentColor(infogChartColor);
    
    // Re-render chart instances to pick up new colors
    if (infogCachedData) {
      if (infogCurrentLayout === 1) {
        renderInfogTrendChart(infogCachedData.trend || {}, 'infog-chart-trend');
        renderInfogImportanceChart(infogCachedData.comparison?.feature_importance || {}, 'infog-chart-importance');
      } else if (infogCurrentLayout === 2) {
        if (infogCachedData.eda_highlights?.scatter_data) {
          renderInfogHistogramChart(infogCachedData.eda_highlights.scatter_data, infogCachedData.overview?.target || '', 'infog-chart-hist-2');
        }
      } else if (infogCurrentLayout === 3) {
        renderInfogImportanceChart(infogCachedData.comparison?.feature_importance || {}, 'infog-chart-importance-3');
        if (infogCachedData.eda_highlights?.scatter_data && infogCachedData.eda_highlights?.top_correlations?.length > 0) {
          const topFeat = infogCachedData.eda_highlights.top_correlations[0].feature;
          renderInfogScatterChart(infogCachedData.eda_highlights.scatter_data, topFeat, infogCachedData.overview?.target || '', 'infog-chart-scatter-3');
        }
      }
    }
  }

  bgInput?.addEventListener('input', applyColors);
  shapeInput?.addEventListener('input', applyColors);
  chartInput?.addEventListener('input', applyColors);
  textInput?.addEventListener('input', applyColors);

  // Layout Toggle Handlers
  document.querySelectorAll('.infog-layout-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.infog-layout-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.dataset.layout;
      
      // Hide all layouts
      document.querySelectorAll('.infog-layout').forEach(l => l.classList.remove('active'));
      // Show targeted layout
      document.getElementById('infog-layout-' + targetId)?.classList.add('active');
      
      infogCurrentLayout = parseInt(targetId);

      // Re-render charts for specific layout
      if (infogCachedData) {
        if (infogCurrentLayout === 1) {
          renderInfogTrendChart(infogCachedData.trend || {}, 'infog-chart-trend');
          renderInfogImportanceChart(infogCachedData.comparison?.feature_importance || {}, 'infog-chart-importance');
        } else if (infogCurrentLayout === 2) {
          if (infogCachedData.eda_highlights?.scatter_data) {
            renderInfogHistogramChart(infogCachedData.eda_highlights.scatter_data, infogCachedData.overview?.target || '', 'infog-chart-hist-2');
          }
        } else if (infogCurrentLayout === 3) {
          renderInfogImportanceChart(infogCachedData.comparison?.feature_importance || {}, 'infog-chart-importance-3');
          if (infogCachedData.eda_highlights?.scatter_data && infogCachedData.eda_highlights?.top_correlations?.length > 0) {
            const topFeat = infogCachedData.eda_highlights.top_correlations[0].feature;
            renderInfogScatterChart(infogCachedData.eda_highlights.scatter_data, topFeat, infogCachedData.overview?.target || '', 'infog-chart-scatter-3');
          }
        }
      }
    });
  });

  // Preset Handlers
  document.getElementById('infog-btn-preset-dark')?.addEventListener('click', (e) => {
    document.querySelectorAll('.infog-theme-btn').forEach(b => b.classList.remove('active'));
    e.currentTarget.classList.add('active');
    bgInput.value = '#0d1225';
    shapeInput.value = '#1e293b';
    chartInput.value = '#8b5cf6';
    textInput.value = '#f8fafc';
    infogCurrentTheme = 'dark';
    canvas.classList.remove('light');
    applyColors();
  });

  document.getElementById('infog-btn-preset-light')?.addEventListener('click', (e) => {
    document.querySelectorAll('.infog-theme-btn').forEach(b => b.classList.remove('active'));
    e.currentTarget.classList.add('active');
    bgInput.value = '#f9fafb';
    shapeInput.value = '#ffffff';
    chartInput.value = '#d4af37';
    textInput.value = '#0f172a';
    infogCurrentTheme = 'light';
    canvas.classList.add('light');
    applyColors();
  });
  
  // Initial apply
  applyColors();
}

function applyInfographicAccentColor(color) {
  const canvas = document.getElementById('infographic-canvas');
  if (!canvas) return;

  // Section titles
  canvas.querySelectorAll('.infog-section-title').forEach(el => {
    el.style.color = color;
  });

  // Logo icon background
  const logoIcon = canvas.querySelector('.infog-logo-icon');
  if (logoIcon) logoIcon.style.background = `linear-gradient(135deg, ${color}, ${color}cc)`;

  // R2 ring stroke
  const r2Circle = document.getElementById('infog-r2-circle');
  if (r2Circle) r2Circle.setAttribute('stroke', color);

  // Narrative left borders
  canvas.querySelectorAll('.infog-narrative-item').forEach(el => {
    el.style.borderLeftColor = color;
  });

  // Narrative number badges
  canvas.querySelectorAll('.infog-narrative-num').forEach(el => {
    el.style.background = `linear-gradient(135deg, ${color}, ${color}cc)`;
  });

  // Descriptive Stats left borders
  canvas.querySelectorAll('.infog-desc-item').forEach(el => {
    el.style.borderLeftColor = color;
  });

  // Footer branding
  const footerStrong = canvas.querySelector('.infog-footer strong');
  if (footerStrong) footerStrong.style.color = color;
}

// Initialize toolbar on page load
document.addEventListener('DOMContentLoaded', () => {
  initInfographicToolbar();
});

async function loadInfographic() {
  try {
    const res = await fetchAPI(API.infographic);
    if (res.status !== 'success') {
      showToast('Gagal memuat data infografis', 'error');
      return;
    }
    const d = res.data;
    infogCachedData = d;

    // --- Header ---
    const now = new Date(d.generated_at);
    const dateStr = now.toLocaleDateString('id-ID', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    document.getElementById('infog-date').textContent = dateStr;
    document.getElementById('infog-footer-date').textContent = dateStr;
    // Helper to safely set text
    function setText(id, text) {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    }

    // --- Overview Stats ---
    setText('infog-rows', d.overview.rows);
    setText('infog-cols', d.overview.cols);
    setText('infog-features', d.overview.features.length);
    setText('infog-target', d.overview.target);

    // --- Model Metrics ---
    const m = d.metrics || {};
    setText('infog-model-name', m.model_name || '—');
    const r2 = m.r2_score || 0;
    setText('infog-r2-val', r2);
    setText('infog-mae-val', m.mae || '—');
    setText('infog-rmse-val', m.rmse || '—');

    // Animate R2 ring
    const circumference = 2 * Math.PI * 35; // r=35
    const r2Pct = Math.max(0, Math.min(1, r2));
    const offset = circumference - (r2Pct * circumference);
    const circle = document.getElementById('infog-r2-circle');
    if (circle) {
      circle.setAttribute('stroke-dasharray', circumference);
      setTimeout(() => {
        circle.style.transition = 'stroke-dashoffset 1.5s ease';
        circle.setAttribute('stroke-dashoffset', offset);
      }, 200);
    }

    // --- Overview Stats Layout 2 ---
    const totalData = d.overview.rows || 0;
    setText('infog-l2-rows', totalData.toLocaleString('id-ID'));
    setText('infog-l2-target', d.overview.target || '—');
    setText('infog-l2-best-model', m.model_name || '—');

    // --- Overview Stats Layout 3 ---
    setText('infog-l3-r2', r2);
    setText('infog-l3-rmse', m.rmse || '—');
    setText('infog-l3-mae', m.mae || '—');

    // --- Comparison Bars (Layout 1) ---
    const comp = d.comparison || {};
    const lrR2 = comp.lr_metrics?.r2_score || 0;
    const rfR2 = comp.rf_metrics?.r2_score || 0;
    setText('infog-lr-r2', lrR2);
    setText('infog-rf-r2', rfR2);
    setTimeout(() => {
      const lrBar = document.getElementById('infog-lr-bar');
      const rfBar = document.getElementById('infog-rf-bar');
      if (lrBar) lrBar.style.width = Math.max(0, lrR2 * 100) + '%';
      if (rfBar) rfBar.style.width = Math.max(0, rfR2 * 100) + '%';
    }, 300);

    // --- Trend ---
    const trend = d.trend || {};
    setText('infog-trend-label', trend.target_name || 'Target');
    setText('infog-trend-latest', trend.latest_value ?? '—');
    setText('infog-trend-avg', trend.avg_value ?? '—');
    setText('infog-trend-min', trend.min_value ?? '—');
    setText('infog-trend-max', trend.max_value ?? '—');

    // Trend chart
    renderInfogTrendChart(trend);

    // --- Correlations ---
    const corrContainer = document.getElementById('infog-correlations');
    const topCorr = (d.eda_highlights?.top_correlations || []).slice(0, 3);
    const rankClasses = ['gold', 'silver', 'bronze', 'default', 'default'];
    const corrHtml = topCorr.map((c, i) => {
      const val = c.correlation;
      const absVal = Math.abs(val);
      const colorClass = val >= 0 ? 'positive' : 'negative';
      const barColor = val >= 0 ? '#22c55e' : '#ef4444';
      return `
        <div class="infog-corr-item">
          <div class="infog-corr-rank ${rankClasses[i] || 'default'}">${i + 1}</div>
          <div class="infog-corr-name">${c.feature}</div>
          <div class="infog-corr-bar-track">
            <div class="infog-corr-bar-fill" style="width:${absVal * 100}%; background:${barColor};"></div>
          </div>
          <div class="infog-corr-val ${colorClass}">${val > 0 ? '+' : ''}${val}</div>
        </div>
      `;
    }).join('');
    
    if (corrContainer) corrContainer.innerHTML = corrHtml;

    // --- Feature Importance Chart ---
    const fi = comp.feature_importance || {};

    // --- Dynamic Contextual Insights & Narratives ---
    const rawInsights = d.eda_highlights?.insights || [];
    const rawNarratives = d.narratives || [];
    const baseTarget = d.overview.target || 'Target';
    const baseModel = m.model_name || 'Machine Learning';

    function buildInsight(idx, fallback) {
      const ins = rawInsights[idx];
      if (!ins) return `<div class="infog-insight-card"><div class="infog-insight-icon"><i data-lucide="zap"></i></div><div class="infog-insight-text">${fallback}</div></div>`;
      let text = typeof ins === 'string' ? ins : `<strong>${ins.feature}</strong>: ${ins.insight}`;
      return `<div class="infog-insight-card"><div class="infog-insight-icon"><i data-lucide="zap"></i></div><div class="infog-insight-text">${text}</div></div>`;
    }

    function buildNarrative(sentences) {
      return sentences.map((n, i) => `
        <div class="infog-narrative-item">
          <div class="infog-narrative-num">${i + 1}</div>
          <span>${n}</span>
        </div>
      `).join('');
    }

    // Layout 1: Executive Summary
    const l1Narratives = [
      `Analisis prediktif dijalankan atas ${totalData.toLocaleString('id-ID')} baris data historis untuk memproyeksikan target utama: <strong>${baseTarget}</strong>.`,
      `Algoritma <strong>${baseModel}</strong> berhasil terpilih sebagai model dengan performa paling akurat dan efisien.`
    ];
    document.getElementById('infog-l1-narratives').innerHTML = buildNarrative(l1Narratives);
    document.getElementById('infog-l1-insights').innerHTML = buildInsight(0, 'Data menunjukkan pola musiman yang konsisten dengan pertumbuhan yang diprediksi akan stabil dalam kuartal mendatang.');

    // Layout 2 & 3: Statistik Deskriptif (Replacing Narratives)
    const descContainer2 = document.getElementById('infog-l2-desc-stats');
    const descContainer3 = document.getElementById('infog-l3-desc-stats');
    
    // Pick top 4 features from top_correlations, or fallback to first 4 features
    let descFeatures = [];
    if (topCorr && topCorr.length > 0) {
      descFeatures = topCorr.slice(0, 4).map(c => c.feature);
    } else {
      descFeatures = d.overview.features ? d.overview.features.slice(0, 4) : [];
    }

    const descStatsData = d.eda_highlights?.descriptive_summary || [];
    const descStatsHtml = descFeatures.map(feat => {
      const stat = descStatsData.find(s => s.feature === feat) || {};
      const mean = stat.mean !== undefined ? stat.mean : '-';
      const min = stat.min !== undefined ? stat.min : '-';
      const max = stat.max !== undefined ? stat.max : '-';
      
      return `
        <div class="infog-desc-item">
          <div class="infog-desc-feature" title="${feat}">${feat}</div>
          <div class="infog-desc-metrics">
            <div class="infog-desc-metric">
              <span class="infog-desc-metric-label">Min</span>
              <span class="infog-desc-metric-value">${min}</span>
            </div>
            <div class="infog-desc-metric">
              <span class="infog-desc-metric-label">Mean</span>
              <span class="infog-desc-metric-value">${mean}</span>
            </div>
            <div class="infog-desc-metric">
              <span class="infog-desc-metric-label">Max</span>
              <span class="infog-desc-metric-value">${max}</span>
            </div>
          </div>
        </div>
      `;
    }).join('');

    if (descContainer2) descContainer2.innerHTML = descStatsHtml;
    if (descContainer3) descContainer3.innerHTML = descStatsHtml;

    // Layout 2 & 3: Insights
    document.getElementById('infog-l2-insights').innerHTML = buildInsight(1, 'Intervensi pada faktor-faktor dengan korelasi tertinggi dapat mempercepat pencapaian target efisiensi bisnis.');
    
    setText('infog-model-name-3', baseModel);
    
    const l3InsightsEl = document.getElementById('infog-l3-insights');
    if (l3InsightsEl) l3InsightsEl.innerHTML = buildInsight(2, 'Model siap dideploy. Disarankan menjadwalkan retraining berkala jika terjadi data drift pada variabel independen utama.');

    // Layout 3 Correlations (Reuse layout 1 logic)
    const corrContainer3 = document.getElementById('infog-l3-correlations');
    if (corrContainer3) {
      corrContainer3.innerHTML = topCorr.map((c, i) => {
        const val = c.correlation;
        const colorClass = val >= 0 ? 'positive' : 'negative';
        const barColor = val >= 0 ? '#22c55e' : '#ef4444';
        return `
          <div class="infog-corr-item">
            <div class="infog-corr-name">${c.feature}</div>
            <div class="infog-corr-val ${colorClass}">${val > 0 ? '+' : ''}${val}</div>
            <div class="infog-corr-bar-track" style="grid-column:1/-1; margin-top:8px;">
              <div class="infog-corr-bar-fill" style="width:${Math.abs(val) * 100}%; background:${barColor};"></div>
            </div>
          </div>
        `;
      }).join('');
    }

    // Re-render icons
    if (typeof lucide !== 'undefined') lucide.createIcons();
    
    // Apply initial accent
    applyInfographicAccentColor(infogChartColor);

    // Initial Chart Render based on active layout
    if (infogCurrentLayout === 1) {
        renderInfogTrendChart(trend, 'infog-chart-trend');
        renderInfogImportanceChart(fi, 'infog-chart-importance');
    } else if (infogCurrentLayout === 2) {
        if (d.eda_highlights?.scatter_data) {
          renderInfogHistogramChart(d.eda_highlights.scatter_data, d.overview?.target || '', 'infog-chart-hist-2');
        }
    } else if (infogCurrentLayout === 3) {
        renderInfogImportanceChart(fi, 'infog-chart-importance-3');
        if (d.eda_highlights?.scatter_data && d.eda_highlights?.top_correlations?.length > 0) {
          const topFeat = d.eda_highlights.top_correlations[0].feature;
          renderInfogScatterChart(d.eda_highlights.scatter_data, topFeat, d.overview?.target || '', 'infog-chart-scatter-3');
        }
    }

  } catch (err) {
    console.error('Infographic error:', err);
    showToast('Gagal memuat infografis: ' + err.message, 'error');
  }
}

function renderInfogTrendChart(trend, canvasId = 'infog-chart-trend') {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (infogChartTrend) infogChartTrend.destroy();

  const forecast = trend.forecast;
  const baseLabels = trend.labels || [];
  const baseData = trend.target_data || [];
  const hasForecast = forecast && Array.isArray(forecast.target_data) && forecast.target_data.length > 0;
  const labels = hasForecast ? baseLabels.concat(forecast.labels || []) : baseLabels;
  const actualSeries = hasForecast ? baseData.concat(new Array(forecast.target_data.length).fill(null)) : baseData;
  const forecastSeries = hasForecast && baseData.length
    ? new Array(baseData.length - 1).fill(null).concat([baseData[baseData.length - 1]], forecast.target_data)
    : [];
  const step = Math.max(1, Math.floor(labels.length / 8));
  
  const isLight = infogCurrentTheme === 'light';
  const gridColor = isLight ? 'rgba(0,0,0,0.03)' : 'rgba(255,255,255,0.03)';
  const tickColor = isLight ? '#94a3b8' : '#94a3b8';
  const tooltipBg = isLight ? 'rgba(255,255,255,0.98)' : 'rgba(15,23,42,0.9)';
  const tooltipText = isLight ? '#0f172a' : '#f8fafc';

  infogChartTrend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Aktual',
        data: actualSeries,
        borderColor: infogChartColor,
        backgroundColor: createGradient(ctx, infogChartColor),
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
      }].concat(hasForecast ? [{
        label: 'Prediksi',
        data: forecastSeries,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245,158,11,0.08)',
        borderWidth: 2,
        borderDash: [5, 4],
        fill: false,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 4,
        spanGaps: true
      }] : [])
    },
    options: { animation: false,
      animation: false,
      responsive: true, maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: hasForecast, position: 'bottom', labels: { usePointStyle: true, font: { size: 9 } } },
        tooltip: { 
          backgroundColor: tooltipBg, titleColor: tooltipText, bodyColor: tooltipText,
          borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)', borderWidth: 1, 
          cornerRadius: 8, titleFont: { size: 11 }, bodyFont: { size: 11 } 
        }
      },
      scales: {
        x: { ticks: { color: tickColor, callback: (val, i) => i % step === 0 ? labels[i] : '', maxRotation: 0, font: { size: 9 } }, grid: { display: false } },
        y: { ticks: { color: tickColor, font: { size: 9 }, callback: v => v + '%' }, grid: { color: gridColor } }
      }
    }
  });
}

function renderInfogImportanceChart(importance, canvasId = 'infog-chart-importance') {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (infogChartImportance) infogChartImportance.destroy();

  const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);
  const labels = sorted.map(e => e[0]);
  const values = sorted.map(e => e[1]);
  const colors = ['#14b8a6', '#3b82f6', '#8b5cf6', '#f59e0b', '#ec4899', '#22c55e', '#ef4444', '#6366f1', '#06b6d4'];

  const isLight = infogCurrentTheme === 'light';
  const gridColor = isLight ? 'rgba(0,0,0,0.03)' : 'rgba(255,255,255,0.03)';
  const tickColor = isLight ? '#94a3b8' : '#94a3b8';
  const tooltipBg = isLight ? 'rgba(255,255,255,0.98)' : 'rgba(15,23,42,0.9)';
  const tooltipText = isLight ? '#0f172a' : '#f8fafc';

  infogChartImportance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map((_, i) => colors[i % colors.length]),
        borderRadius: 4,
        barThickness: 14
      }]
    },
    options: { animation: false,
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { 
        legend: { display: false }, 
        tooltip: { 
          backgroundColor: tooltipBg, titleColor: tooltipText, bodyColor: tooltipText,
          borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)', borderWidth: 1, 
          cornerRadius: 8 
        } 
      },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 9 } } },
        y: { grid: { display: false }, ticks: { color: tickColor, font: { size: 10 } } }
      }
    }
  });
}

function renderInfogHistogramChart(records, col, canvasId = 'infog-chart-hist-2') {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (infogChartHistogram) infogChartHistogram.destroy();

  const toNumericValues = values => values.map(v => Number(v)).filter(v => Number.isFinite(v));
  const actualData = toNumericValues((records || []).map(d => d[col]));
  if (actualData.length === 0) return;

  const forecast = infogCachedData?.trend?.forecast || infogCachedData?.eda_highlights?.forecast;
  const targetName = infogCachedData?.overview?.target || forecast?.target_name || col;
  let predictedData = [];

  if (forecast) {
    if (col === targetName) {
      predictedData = toNumericValues(forecast.target_data || []);
    } else if (forecast.feature_data && forecast.feature_data[col]) {
      predictedData = toNumericValues(forecast.feature_data[col]);
    }
  }

  const allValues = actualData.concat(predictedData);
  let min = Math.min(...allValues);
  let max = Math.max(...allValues);
  if (min === max) {
    min -= 1;
    max += 1;
  }

  const bins = 8;
  const step = (max - min) / bins;
  const labels = Array.from({ length: bins }, (_, i) => `${(min + i * step).toFixed(1)}-${(min + (i + 1) * step).toFixed(1)}`);
  const buildHist = values => {
    const hist = new Array(bins).fill(0);
    values.forEach(value => {
      let idx = Math.floor((value - min) / step);
      if (idx < 0) idx = 0;
      if (idx >= bins) idx = bins - 1;
      hist[idx]++;
    });
    return hist;
  };

  const isLight = infogCurrentTheme === 'light';
  const gridColor = isLight ? 'rgba(0,0,0,0.03)' : 'rgba(255,255,255,0.03)';
  const tickColor = '#94a3b8';
  const tooltipBg = isLight ? 'rgba(255,255,255,0.98)' : 'rgba(15,23,42,0.9)';
  const tooltipText = isLight ? '#0f172a' : '#f8fafc';

  infogChartHistogram = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Aktual',
        data: buildHist(actualData),
        backgroundColor: infogChartColor,
        borderRadius: 4
      }].concat(predictedData.length ? [{
        label: 'Prediksi',
        data: buildHist(predictedData),
        backgroundColor: '#f59e0b',
        borderRadius: 4
      }] : [])
    },
    options: { animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: predictedData.length > 0, position: 'bottom', labels: { usePointStyle: true, font: { size: 9 } } },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipText,
          bodyColor: tooltipText,
          borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          cornerRadius: 8
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: tickColor, font: { size: 8 }, maxRotation: 0 } },
        y: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 9 }, precision: 0 } }
      }
    }
  });
}

function renderInfogScatterChart(records, xCol, yCol, canvasId = 'infog-chart-scatter-3') {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (infogChartScatter) infogChartScatter.destroy();

  const data = (records || [])
    .map(d => ({ x: Number(d[xCol]), y: Number(d[yCol]) }))
    .filter(d => Number.isFinite(d.x) && Number.isFinite(d.y));

  const isLight = infogCurrentTheme === 'light';
  const gridColor = isLight ? 'rgba(0,0,0,0.03)' : 'rgba(255,255,255,0.03)';
  const tickColor = '#94a3b8';
  const tooltipBg = isLight ? 'rgba(255,255,255,0.98)' : 'rgba(15,23,42,0.9)';
  const tooltipText = isLight ? '#0f172a' : '#f8fafc';

  infogChartScatter = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: `${xCol} vs ${yCol}`,
        data,
        backgroundColor: infogChartColor,
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    },
    options: { animation: false,
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipText,
          bodyColor: tooltipText,
          borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          cornerRadius: 8
        }
      },
      scales: {
        x: { title: { display: true, text: xCol, color: tickColor, font: { size: 9 } }, grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 9 } } },
        y: { title: { display: true, text: yCol, color: tickColor, font: { size: 9 } }, grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 9 } } }
      }
    }
  });
}
