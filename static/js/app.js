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
  trend: '/api/trend',
  predict: '/api/predict',
  history: '/api/history',
  upload: '/api/upload-csv',
  export: '/api/export',
  clearHistory: '/api/clear-history'
};

// ---- State Aplikasi ----
let chartGDP = null;
let chartIndicators = null;
let chartActualVsPred = null;
let chartCoefficients = null;

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
    case 'prediksi': loadHistory(); break;
    case 'model': loadModelInfo(); break;
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
        label: 'GDP Growth (%)',
        data: data.gdp_growth,
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

  chartIndicators = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [
        { label: 'Inflasi (%)', data: data.inflasi, borderColor: '#f59e0b', borderWidth: 2, tension: 0.4, pointRadius: 0 },
        { label: 'Pengangguran (%)', data: data.pengangguran, borderColor: '#ef4444', borderWidth: 2, tension: 0.4, pointRadius: 0 },
        { label: 'Suku Bunga (%)', data: data.suku_bunga, borderColor: '#14b8a6', borderWidth: 2, tension: 0.4, pointRadius: 0 }
      ]
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
  formData.forEach((val, key) => { input[key] = parseFloat(val); });

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
  tbody.innerHTML = reversed.map((h, i) => `
    <tr>
      <td>${history.length - i}</td>
      <td>${h.timestamp}</td>
      <td>${h.input.Tahun || '—'}</td>
      <td>Q${h.input.Kuartal || '—'}</td>
      <td>${h.input.Inflasi_Persen || '—'}%</td>
      <td><strong style="color:#8b5cf6">${h.prediction}%</strong></td>
    </tr>
  `).join('');
}

// ===========================================================
//                    TENTANG MODEL
// ===========================================================
async function loadModelInfo() {
  try {
    const res = await fetchAPI(API.modelInfo);
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

    // Intercept
    document.getElementById('intercept-value').textContent = data.intercept;

    // Actual vs Predicted chart
    renderActualVsPredChart(data.actual_vs_predicted);

    // Coefficients chart
    renderCoefficientsChart(data.coefficients);

  } catch (err) {
    console.error('Model info error:', err);
    showToast('Gagal memuat info model', 'error');
  }
}

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

    // Statistics table
    renderStatsTable(info.statistics);

  } catch (err) {
    console.error('Dataset error:', err);
    showToast('Gagal memuat dataset', 'error');
  }
}

function renderDataTable(columns, data) {
  const thead = document.getElementById('dataset-thead');
  const tbody = document.getElementById('dataset-tbody');

  thead.innerHTML = '<tr>' + columns.map(c => `<th>${c}</th>`).join('') + '</tr>';
  tbody.innerHTML = data.map(row =>
    '<tr>' + columns.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>'
  ).join('');

  // Store for search
  window._datasetColumns = columns;
  window._datasetData = data;
}

function renderStatsTable(statistics) {
  const thead = document.getElementById('stats-thead');
  const tbody = document.getElementById('stats-tbody');

  const statKeys = Object.keys(statistics);
  if (statKeys.length === 0) return;

  const metrics = Object.keys(statistics[statKeys[0]]);

  thead.innerHTML = '<tr><th>Statistik</th>' + statKeys.map(k => `<th>${k}</th>`).join('') + '</tr>';
  tbody.innerHTML = metrics.map(m =>
    '<tr><td><strong>' + m + '</strong></td>' +
    statKeys.map(k => `<td>${parseFloat(statistics[k][m]).toFixed(2)}</td>`).join('') +
    '</tr>'
  ).join('');
}

function initDatasetSearch() {
  const input = document.getElementById('dataset-search');
  input?.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    if (!window._datasetData) return;

    const filtered = window._datasetData.filter(row =>
      Object.values(row).some(v => String(v).toLowerCase().includes(q))
    );
    const tbody = document.getElementById('dataset-tbody');
    const cols = window._datasetColumns;
    tbody.innerHTML = filtered.map(row =>
      '<tr>' + cols.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>'
    ).join('');
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
