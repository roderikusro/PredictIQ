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
  clearHistory: '/api/clear-history',
  eda: '/api/eda'
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
      document.getElementById('dashboard-timestamp').textContent = dashRes.timestamp || '';

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
      renderGDPChart(trendRes.data);
      renderIndicatorsChart(trendRes.data);
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

  // Sampling labels agar tidak terlalu padat
  const step = Math.max(1, Math.floor(data.labels.length / 12));
  const displayLabels = data.labels.map((l, i) => i % step === 0 ? l : '');

  chartGDP = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [{
        label: data.target_name || 'Target',
        data: data.target_data,
        borderColor: '#8b5cf6',
        backgroundColor: createGradient(ctx, '#8b5cf6'),
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: '#8b5cf6',
        pointBorderColor: '#0a0e1a',
        pointBorderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
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

function renderIndicatorsChart(data) {
  const ctx = document.getElementById('chart-indicators');
  if (!ctx) return;
  if (chartIndicators) chartIndicators.destroy();

  const colors = ['#f59e0b', '#ef4444', '#14b8a6', '#3b82f6', '#ec4899'];
  let datasets = [];
  
  if (data.indicators) {
    let i = 0;
    for (const [key, values] of Object.entries(data.indicators)) {
      datasets.push({
        label: key,
        data: values,
        borderColor: colors[i % colors.length],
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0
      });
      i++;
    }
  }

  chartIndicators = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: datasets
    },
    options: {
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

    // Overview cards
    document.getElementById('info-model-type').textContent = data.model_type;
    document.getElementById('info-library').textContent = data.library;
    document.getElementById('info-scaler').textContent = data.scaler;
    document.getElementById('info-split').textContent = `${data.train_split} / ${data.test_split}`;

    // Metrics bars
    const mae = data.metrics.mae;
    const mse = data.metrics.mse;
    const r2 = data.metrics.r2_score;

    document.getElementById('metric-mae-val').textContent = mae;
    document.getElementById('metric-mse-val').textContent = mse;
    document.getElementById('metric-r2-val').textContent = r2;

    // Bar fills (normalize for visual)
    setTimeout(() => {
      document.getElementById('bar-mae').style.width = Math.min(100, mae * 20) + '%';
      document.getElementById('bar-mse').style.width = Math.min(100, mse * 10) + '%';
      document.getElementById('bar-r2').style.width = Math.max(0, r2 * 100) + '%';
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
    renderActualVsPredChart(data.actual_vs_predicted);

    // Toggle description content and chart title
    const descLr = document.getElementById('desc-lr');
    const descRf = document.getElementById('desc-rf');
    const chartTitle = document.getElementById('coef-chart-title');

    if (modelType === 'random_forest') {
      if (descLr) descLr.style.display = 'none';
      if (descRf) descRf.style.display = 'block';
      if (chartTitle) chartTitle.textContent = 'Feature Importance (Tingkat Kepentingan)';
    } else {
      if (descLr) descLr.style.display = 'block';
      if (descRf) descRf.style.display = 'none';
      if (chartTitle) chartTitle.textContent = 'Koefisien Model (Bobot Fitur)';
    }

    // Coefficients / Feature Importance chart
    if (data.coefficients) {
      renderCoefficientsChart(data.coefficients);
    } else if (data.feature_importance) {
      renderCoefficientsChart(data.feature_importance);
    }

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
    options: {
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
    options: {
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
}

document.getElementById('config-model-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const target = document.getElementById('config-target').value;
  const featureCheckboxes = document.querySelectorAll('input[name="config-features"]:checked');
  const features = Array.from(featureCheckboxes).map(cb => cb.value);

  const time_cols = {
    year: document.getElementById('config-time-year')?.value || null,
    quarter: document.getElementById('config-time-quarter')?.value || null,
    month: document.getElementById('config-time-month')?.value || null
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
const _dsRowsPerPage = 5;

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
//                    UPLOAD CSV
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
  document.getElementById('btn-export-history')?.addEventListener('click', () => {
    window.location.href = API.export;
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

// Refresh button
document.getElementById('btn-refresh-trend')?.addEventListener('click', loadDashboard);

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

    // Badges & winner highlight
    const lrCard = document.getElementById('comp-card-lr');
    const rfCard = document.getElementById('comp-card-rf');
    lrCard.classList.remove('winner');
    rfCard.classList.remove('winner');

    if (d.best_model === 'Random Forest') {
      rfCard.classList.add('winner');
      document.getElementById('badge-rf').textContent = 'Best';
      document.getElementById('badge-rf').style.background = 'rgba(245,158,11,0.2)';
      document.getElementById('badge-rf').style.color = '#f59e0b';
      document.getElementById('badge-lr').textContent = '';
    } else {
      lrCard.classList.add('winner');
      document.getElementById('badge-lr').textContent = 'Best';
      document.getElementById('badge-lr').style.background = 'rgba(245,158,11,0.2)';
      document.getElementById('badge-lr').style.color = '#f59e0b';
      document.getElementById('badge-rf').textContent = '';
    }

    // Charts
    renderCompMetricsChart(lr, rf);
    renderFeatureImportanceChart(d.random_forest.feature_importance);
    renderCompAVPChart(d.linear_regression.actual_vs_predicted, d.random_forest.actual_vs_predicted);

    lucide.createIcons();
  } catch (err) {
    console.error('Comparison error:', err);
    showToast('Gagal memuat perbandingan model', 'error');
  }
}

function renderCompMetricsChart(lr, rf) {
  const ctx = document.getElementById('chart-comparison-metrics');
  if (!ctx) return;
  if (chartCompMetrics) chartCompMetrics.destroy();

  chartCompMetrics = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['MAE', 'RMSE', 'R² Score'],
      datasets: [
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
      ]
    },
    options: {
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
    options: {
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

function renderCompAVPChart(lrAVP, rfAVP) {
  const ctx = document.getElementById('chart-comparison-avp');
  if (!ctx) return;
  if (chartCompAVP) chartCompAVP.destroy();

  const labels = lrAVP.actual.map((_, i) => `Test ${i + 1}`);

  chartCompAVP = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Aktual', data: lrAVP.actual, backgroundColor: 'rgba(148,163,184,0.5)', borderRadius: 6 },
        { label: 'LR Prediksi', data: lrAVP.predicted, backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 6 },
        { label: 'RF Prediksi', data: rfAVP.predicted, backgroundColor: 'rgba(20,184,166,0.7)', borderRadius: 6 }
      ]
    },
    options: {
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
    const res = await fetchAPI(API.eda);
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

  const data = edaDataCache.scatter_data.map(d => d[col]);
  
  // Create simple histogram using bar chart
  const min = Math.min(...data);
  const max = Math.max(...data);
  const bins = 10;
  const step = (max - min) / bins;
  const hist = new Array(bins).fill(0);
  const labels = [];
  
  for(let i=0; i<bins; i++) {
    labels.push(`${(min + i*step).toFixed(1)} - ${(min + (i+1)*step).toFixed(1)}`);
  }

  data.forEach(val => {
    let idx = Math.floor((val - min) / step);
    if (idx >= bins) idx = bins - 1;
    hist[idx]++;
  });

  chartEdaDist = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: `Distribusi ${col}`,
        data: hist,
        backgroundColor: 'rgba(139, 92, 246, 0.7)',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
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
    options: {
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
