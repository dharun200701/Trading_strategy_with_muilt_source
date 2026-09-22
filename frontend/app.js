const API_BASE_URL = 'http://127.0.0.1:8001';
const API = {
  health: `${API_BASE_URL}/api/health`,
  latest: `${API_BASE_URL}/api/market/latest`,
  history: `${API_BASE_URL}/api/market/history`,
  prediction: `${API_BASE_URL}/api/prediction/latest`,
  risk: `${API_BASE_URL}/api/risk/latest`,
  news: `${API_BASE_URL}/api/news/hdfcbank?limit=8`,
};
const JANUARY_XGBOOST_DATA = 'data/january-2026-xgboost.json';

const UNAVAILABLE = 'Unavailable';
const state = { results: {}, errors: {} };

function isValidNumber(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
}

function displayValue(value, formatter = String) {
  if (value === null || value === undefined || value === '' || (typeof value === 'number' && !Number.isFinite(value))) {
    return UNAVAILABLE;
  }
  return formatter(value);
}

function setText(id, value, formatter = String) {
  const element = document.getElementById(id);
  if (element) element.textContent = displayValue(value, formatter);
}

function formatNumber(value, decimals = 2) {
  return isValidNumber(value) ? Number(value).toLocaleString('en-IN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : UNAVAILABLE;
}

function formatCurrency(value) {
  return isValidNumber(value) ? `₹${formatNumber(value)}` : UNAVAILABLE;
}

function formatPercent(value, sourceIsFraction = true) {
  if (!isValidNumber(value)) return UNAVAILABLE;
  const percentage = sourceIsFraction ? Number(value) * 100 : Number(value);
  return `${formatNumber(percentage, 1)}%`;
}

function getPath(object, path) {
  return path.split('.').reduce((current, key) => (current && current[key] !== undefined ? current[key] : undefined), object);
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) {
    let detail = '';
    try {
      const errorBody = await response.json();
      detail = typeof errorBody.detail === 'object' ? errorBody.detail.message : errorBody.detail;
    } catch {
      detail = '';
    }
    throw new Error(detail || `Backend returned HTTP ${response.status}`);
  }
  try {
    return await response.json();
  } catch {
    throw new Error('Backend returned invalid JSON');
  }
}

async function fetchOptional(name, url) {
  try {
    state.results[name] = await fetchJson(url);
    delete state.errors[name];
  } catch (error) {
    state.results[name] = null;
    state.errors[name] = error.message;
    console.error(`API request failed for ${name}:`, error);
  }
}

function statusText(value) {
  if (!value) return UNAVAILABLE;
  return String(value).replaceAll('_', ' ').toUpperCase();
}

function setHealth(id, value, fallback = 'Not reported by backend') {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = statusText(value || fallback);
  element.className = 'health-status';
  if (value) element.classList.add(String(value).toLowerCase());
}

function formatTimeStamp(date = new Date()) {
  return new Date(date).toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

function renderStatus() {
  const latest = state.results.latest;
  const dataAvailable = latest && isValidNumber(latest.close);
  document.querySelector('.live-dot')?.classList.toggle('offline', !dataAvailable);
  const statusLabel = document.getElementById('market-status-label');
  if (statusLabel) {
    statusLabel.textContent = dataAvailable ? 'Market data available' : 'Market data delayed';
  }

  const lastUpdated = document.getElementById('last-updated');
  if (lastUpdated) {
    lastUpdated.textContent = state.lastUpdated ? formatTimeStamp(state.lastUpdated) : '--:--:--';
  }
}

function getDecisionContext() {
  const latest = state.results.latest;
  const prediction = state.results.prediction;
  const risk = state.results.risk;
  const currentPrice = isValidNumber(latest?.close) ? Number(latest.close) : null;
  const stopDistance = isValidNumber(risk?.stop_loss_distance) ? Number(risk.stop_loss_distance) : null;
  const confidence = isValidNumber(prediction?.confidence) ? Number(prediction.confidence) : 0;
  const signal = (prediction?.signal || prediction?.prediction || 'HOLD').toUpperCase();
  const action = signal === 'BUY' || signal === 'SELL' ? signal : 'HOLD';

  let entry = 'Not available';
  let target = 'Not available';
  let stop = 'Not available';
  let move = 'Not available';
  let subtitle = 'Awaiting market data';

  if (currentPrice !== null && stopDistance !== null) {
    if (action === 'BUY') {
      entry = `₹${formatNumber(currentPrice, 2)}`;
      const targetPrice = currentPrice * (1 + stopDistance * 2.2);
      const stopPrice = currentPrice * (1 - stopDistance);
      target = `₹${formatNumber(targetPrice, 2)}`;
      stop = `₹${formatNumber(stopPrice, 2)}`;
      move = `${formatNumber(((targetPrice - currentPrice) / currentPrice) * 100, 1)}%`;
      subtitle = 'Model-based signal';
    } else if (action === 'SELL') {
      entry = `₹${formatNumber(currentPrice, 2)}`;
      const targetPrice = currentPrice * (1 - stopDistance * 2.2);
      const stopPrice = currentPrice * (1 + stopDistance);
      target = `₹${formatNumber(targetPrice, 2)}`;
      stop = `₹${formatNumber(stopPrice, 2)}`;
      move = `${formatNumber(((currentPrice - targetPrice) / currentPrice) * 100, 1)}%`;
      subtitle = 'Model-based signal';
    } else {
      entry = `₹${formatNumber(currentPrice, 2)}`;
      const scenarioTarget = currentPrice * (1 + stopDistance * 2.2);
      const scenarioStop = currentPrice * (1 - stopDistance);
      target = `₹${formatNumber(scenarioTarget, 2)}`;
      stop = `₹${formatNumber(scenarioStop, 2)}`;
      move = `${formatNumber(((scenarioTarget - currentPrice) / currentPrice) * 100, 1)}%`;
      subtitle = 'Hold and monitor the model-based scenario levels';
    }
  }

  return {
    currentPrice,
    confidence,
    action,
    entry,
    target,
    stop,
    move,
    subtitle,
    signal,
  };
}

function renderMetrics() {
  const latest = state.results.latest;
  const previousClose = latest?.previous_close;
  const dailyChange = isValidNumber(latest?.change) ? latest.change : (isValidNumber(latest?.close) && isValidNumber(previousClose) ? Number(latest.close) - Number(previousClose) : undefined);
  setText('latest-price', latest?.close, formatCurrency);
  setText('daily-change', dailyChange, value => `${formatCurrency(value)}${isValidNumber(previousClose) ? ` (${formatPercent(Number(value) / Number(previousClose))})` : ''}`);
}

function renderPrediction() {
  const prediction = state.results.prediction;
  const latest = state.results.latest;
  const risk = state.results.risk;
  const decision = getDecisionContext();
  const hasConfidence = isValidNumber(prediction?.confidence);
  const confidence = hasConfidence ? Number(prediction.confidence) : null;
  const confidencePercent = hasConfidence ? Math.min(Math.max(confidence * 100, 0), 100) : 0;

  setText('prediction-detail', prediction?.prediction);
  setText('prediction-badge', decision.action || prediction?.prediction || 'Unavailable');
  setText('signal-detail', prediction?.signal || 'HOLD');
  setText('confidence-value', confidence, value => `${formatNumber(value * 100, 1)}%`);

  const confidenceBar = document.getElementById('prediction-confidence-bar');
  if (confidenceBar) confidenceBar.style.width = `${confidencePercent}%`;

  const currentPrice = isValidNumber(latest?.close) ? Number(latest.close) : null;
  const stopDistance = isValidNumber(risk?.stop_loss_distance) ? Number(risk.stop_loss_distance) : null;
  const direction = prediction?.signal || 'HOLD';
  const stopPrice = currentPrice !== null && stopDistance !== null && (direction === 'BUY' || direction === 'SELL')
    ? currentPrice * (direction === 'SELL' ? 1 + stopDistance : 1 - stopDistance)
    : null;
  setText('reference-price', currentPrice, formatCurrency);
  setText('stop-price', stopPrice, formatCurrency);

  const suggestion = direction === 'BUY'
    ? 'Maintain a buy bias while price remains above the reference zone.'
    : direction === 'SELL'
      ? 'Maintain a sell bias while price remains below the reference zone.'
      : direction === 'HOLD'
        ? 'Current model conviction is not strong enough to justify a directional trade.'
        : 'Suggestion unavailable until market data arrives.';
  setText('suggestion-copy', suggestion);

  const mainAction = document.getElementById('main-action');
  const actionSubtitle = document.getElementById('action-subtitle');
  const decisionBadge = document.getElementById('decision-badge');
  const decisionTitle = document.getElementById('decision-title');

  if (mainAction) mainAction.textContent = decision.action;
  if (actionSubtitle) actionSubtitle.textContent = decision.subtitle;
  if (decisionBadge) decisionBadge.textContent = decision.action === 'HOLD' ? 'Wait / monitor' : 'Model-based signal';
  if (decisionTitle) decisionTitle.textContent = decision.action === 'HOLD' ? 'Current signal' : 'Model signal';

  const decisionEntry = document.getElementById('decision-entry');
  const decisionTarget = document.getElementById('decision-target');
  const decisionStop = document.getElementById('decision-stop');
  const decisionMove = document.getElementById('decision-move');

  if (decisionEntry) decisionEntry.textContent = decision.entry;
  if (decisionTarget) decisionTarget.textContent = decision.target;
  if (decisionStop) decisionStop.textContent = decision.stop;
  if (decisionMove) decisionMove.textContent = decision.move;
}

function renderRisk() {
  const risk = state.results.risk;
  setText('risk-status', risk?.status, statusText);
  setText('position-size', risk?.position_size, value => `${formatNumber(value)} units`);
  setText('risk-per-trade', risk?.account_capital && risk?.risk_per_trade ? risk.account_capital * risk.risk_per_trade : undefined, formatCurrency);
  setText('stop-loss', risk?.stop_loss_distance, formatPercent);
  setText('atr', risk?.atr, formatNumber);
  setText('volatility', risk?.volatility, formatPercent);
}

function renderTradeDecision() {
  const decision = getDecisionContext();
  const risk = state.results.risk;
  const action = document.getElementById('trade-action');
  const actionCard = document.getElementById('trade-action-card');
  const guidance = document.getElementById('trade-guidance');

  if (action) action.textContent = decision.currentPrice === null ? 'Loading' : decision.action;
  if (actionCard) {
    actionCard.classList.remove('buy', 'sell', 'hold', 'loading');
    actionCard.classList.add(decision.currentPrice === null ? 'loading' : decision.action.toLowerCase());
  }

  if (guidance) {
    guidance.textContent = decision.currentPrice === null
      ? 'Reading the latest market and model data...'
      : decision.action === 'BUY'
        ? 'Consider entry near the reference price and protect the position at the stop level.'
        : decision.action === 'SELL'
          ? 'Review an exit near the reference price and respect the protective stop level.'
          : 'No immediate directional trade. Wait for stronger model conviction or a better price setup.';
  }

  setText('trade-reference', decision.currentPrice, formatCurrency);
  setText('trade-target', decision.target === 'Not available' ? undefined : decision.target);
  setText('trade-stop', decision.stop === 'Not available' ? undefined : decision.stop);
  setText('trade-stop-distance', risk?.stop_loss_distance, formatPercent);
  setText('trade-position', risk?.position_size, value => `${formatNumber(value)} units`);
  setText('trade-volatility', risk?.volatility, formatPercent);
}

function getCapitalValues() {
  const latest = state.results.latest;
  const currentPrice = isValidNumber(latest?.close) ? Number(latest.close) : null;
  const percentage = Number(document.getElementById('allocation-percent')?.value || 40);
  const capitalInput = document.getElementById('investment-capital');
  const capital = Number(capitalInput?.value || 50000);
  const allocationValue = capital * (percentage / 100);
  const estimatedShares = currentPrice && currentPrice > 0 ? Math.floor(allocationValue / currentPrice) : 0;
  const usedAmount = currentPrice && currentPrice > 0 ? estimatedShares * currentPrice : 0;
  const remainingCapital = Math.max(capital - usedAmount, 0);

  return {
    capital,
    percentage,
    allocationValue,
    estimatedShares,
    usedAmount,
    remainingCapital,
    currentPrice,
  };
}

function renderCapitalPlan() {
  const capitalInput = document.getElementById('investment-capital');
  const allocationRange = document.getElementById('allocation-percent');
  const allocationValueDisplay = document.getElementById('allocation-percent-value');
  const errorBox = document.getElementById('capital-error');

  const capital = Number(capitalInput?.value || 0);
  const percent = Number(allocationRange?.value || 40);

  if (allocationValueDisplay) allocationValueDisplay.textContent = `${percent}%`;

  const { allocationValue, estimatedShares, usedAmount, remainingCapital, currentPrice } = getCapitalValues();

  if (!capitalInput || Number.isNaN(capital) || capital <= 0) {
    if (errorBox) errorBox.textContent = 'Enter your investment capital.';
    document.getElementById('allocated-capital').textContent = 'Not available';
    document.getElementById('estimated-shares').textContent = 'Not available';
    document.getElementById('amount-used').textContent = 'Not available';
    document.getElementById('remaining-capital').textContent = 'Not available';
    return;
  }

  if (capital < 0) {
    if (errorBox) errorBox.textContent = 'Capital must be greater than ₹0.';
    return;
  }

  if (errorBox) errorBox.textContent = '';

  const allocated = document.getElementById('allocated-capital');
  const shares = document.getElementById('estimated-shares');
  const amountUsed = document.getElementById('amount-used');
  const remaining = document.getElementById('remaining-capital');

  if (allocated) allocated.textContent = `₹${formatNumber(allocationValue, 2)}`;
  if (shares) shares.textContent = `${estimatedShares}`;
  if (amountUsed) amountUsed.textContent = `₹${formatNumber(usedAmount, 2)}`;
  if (remaining) remaining.textContent = `₹${formatNumber(remainingCapital, 2)}`;

  if (currentPrice && currentPrice > 0 && estimatedShares === 0) {
    if (errorBox) errorBox.textContent = 'Calculated quantity is 0 shares at the current price.';
  }
}

function calculateTradePlan() {
  const capitalInput = document.getElementById('investment-capital');
  const planStatus = document.getElementById('plan-status');
  const capital = Number(capitalInput?.value || 0);

  if (!capitalInput || !Number.isFinite(capital) || capital <= 0) {
    if (planStatus) planStatus.textContent = 'Enter an investment amount to calculate your plan.';
    capitalInput?.focus();
    renderCapitalPlan();
    return;
  }

  sessionStorage.setItem('hdfc_investment_capital', String(capital));
  renderCapitalPlan();
  if (planStatus) planStatus.textContent = 'Trade plan updated using your investment amount.';
}

function renderChart() {
  const history = state.results.history?.data;
  const chart = document.getElementById('price-chart');
  if (!Array.isArray(history) || !history.length || typeof Plotly === 'undefined') {
    chart.textContent = 'Historical chart data unavailable';
    return;
  }
  const timeframe = document.querySelector('.timeframe.active')?.dataset.timeframe || '1d';
  const allRows = history.filter(row => row && row.date && isValidNumber(row.close));
  const endDate = allRows.length ? new Date(allRows[allRows.length - 1].date) : new Date();
  const rangeDays = timeframe === '3m' ? 92 : timeframe === '1m' ? 31 : 1;
  const startDate = new Date(endDate);
  startDate.setDate(startDate.getDate() - rangeDays);
  const rows = allRows.filter(row => new Date(row.date) >= startDate);
  if (!rows.length) {
    chart.textContent = 'Historical chart data unavailable';
    return;
  }
  const dates = rows.map(row => row.date);
  const traces = [{ x: dates, y: rows.map(row => Number(row.close)), name: 'Close', line: { color: '#c62828', width: 2 }, type: 'scatter', mode: 'lines' }];
  const optionalSeries = [
    ['sma_20', 'SMA 20', '#7f8891'],
    ['sma_50', 'SMA 50', '#5d646d'],
    ['ema_20', 'EMA 20', '#c62828'],
  ];
  optionalSeries.forEach(([key, name, color]) => {
    if (rows.some(row => isValidNumber(row[key]))) {
      traces.push({ x: dates, y: rows.map(row => isValidNumber(row[key]) ? Number(row[key]) : null), name, line: { color, width: 1.4 }, type: 'scatter', mode: 'lines' });
    }
  });
  Plotly.newPlot(chart, traces, {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#5d646d', family: 'Manrope' },
    margin: { l: 48, r: 18, t: 10, b: 42 }, hovermode: 'x unified', showlegend: true,
    legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 11 } },
    xaxis: { gridcolor: '#e6e0db', zeroline: false }, yaxis: { gridcolor: '#e6e0db', zeroline: false, tickprefix: '₹' },
  }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
}

function selectTimeframe(timeframe) {
  const message = document.getElementById('chart-message');
  if (message) message.textContent = '';
  document.querySelectorAll('.timeframe').forEach(button => button.classList.toggle('active', button.dataset.timeframe === timeframe));
  renderChart();
}

function renderNews() {
  const container = document.getElementById('news-list');
  container.textContent = '';
  const articles = state.results.news?.data;
  if (!Array.isArray(articles) || !articles.length) {
    container.innerHTML = '<p class="unavailable">News unavailable from the backend.</p>';
    return;
  }
  articles.forEach(article => {
    const item = document.createElement('article');
    item.className = 'news-item';
    const headline = document.createElement('h3');
    headline.textContent = article.headline || UNAVAILABLE;
    const meta = document.createElement('p');
    meta.textContent = `${article.news_datetime || article.news_date || UNAVAILABLE} · ${article.source_name || UNAVAILABLE}`;
    const badges = document.createElement('div');
    badges.className = 'news-badges';
    [['sentiment', article.text_sentiment], ['reaction', article.market_reaction], ['impact', isValidNumber(article.impact_score) ? `Impact ${formatNumber(article.impact_score)}` : null]].forEach(([kind, value]) => {
      if (!value) return;
      const badge = document.createElement('span');
      badge.className = `news-badge ${kind} ${String(value).toLowerCase()}`;
      badge.textContent = kind === 'impact' ? value : statusText(value);
      badges.appendChild(badge);
    });
    item.append(headline, meta, badges);
    container.appendChild(item);
  });
}

const DIRECTION_VALUE = { UP: 1, NEUTRAL: 0, DOWN: -1 };

function renderJanuaryPerformance() {
  const rows = state.results.januaryPerformance;
  const message = document.getElementById('performance-message');
  const chart = document.getElementById('january-performance-chart');
  if (!Array.isArray(rows) || !rows.length) {
    if (message) message.textContent = 'Insufficient January 2026 prediction data.';
    if (chart) chart.textContent = 'Insufficient January 2026 prediction data.';
    return;
  }

  const validRows = rows.filter(row => DIRECTION_VALUE[row.actual] !== undefined && DIRECTION_VALUE[row.prediction] !== undefined && row.date);
  const correct = validRows.filter(row => row.actual === row.prediction).length;
  const incorrect = validRows.length - correct;
  const accuracy = validRows.length ? (correct / validRows.length) * 100 : null;

  setText('january-trading-days', validRows.length, value => String(value));
  setText('january-accuracy', accuracy, value => `${formatNumber(value, 2)}%`);
  setText('january-correct', correct, value => String(value));
  setText('january-incorrect', incorrect, value => String(value));

  if (!validRows.length || typeof Plotly === 'undefined') {
    if (message) message.textContent = 'Insufficient January 2026 prediction data.';
    if (chart) chart.textContent = 'Insufficient January 2026 prediction data.';
    return;
  }

  const dates = validRows.map(row => row.date);
  const actual = validRows.map(row => DIRECTION_VALUE[row.actual]);
  const prediction = validRows.map(row => DIRECTION_VALUE[row.prediction]);
  const customData = validRows.map(row => [row.actual, row.prediction, row.actual === row.prediction ? 'Correct Prediction' : 'Incorrect Prediction']);
  const traces = [
    {
      x: dates, y: actual, name: 'Actual Market Movement', type: 'scatter', mode: 'lines+markers',
      line: { color: '#202326', width: 2 }, marker: { color: '#202326', size: 8 },
      customdata: customData,
      hovertemplate: '<b>%{x}</b><br>Actual: %{customdata[0]}<br>XGBoost: %{customdata[1]}<br>Result: %{customdata[2]}<extra></extra>',
    },
    {
      x: dates, y: prediction, name: 'XGBoost Prediction', type: 'scatter', mode: 'lines+markers',
      line: { color: '#a52c32', width: 2, dash: 'dash' }, marker: { color: '#a52c32', size: 8, symbol: 'diamond' },
      customdata: customData,
      hovertemplate: '<b>%{x}</b><br>Actual: %{customdata[0]}<br>XGBoost: %{customdata[1]}<br>Result: %{customdata[2]}<extra></extra>',
    },
  ];

  validRows.forEach((row, index) => {
    if (row.actual !== row.prediction) {
      traces.push({
        x: [row.date, row.date], y: [actual[index], prediction[index]], name: 'Mismatch', type: 'scatter', mode: 'lines',
        line: { color: '#c88c32', width: 3 }, showlegend: false,
        hoverinfo: 'skip',
      });
    }
  });

  Plotly.newPlot(chart, traces, {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#6b6b68', family: 'Manrope' },
    margin: { l: 58, r: 20, t: 18, b: 62 }, hovermode: 'x unified', showlegend: true,
    legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 11 } },
    xaxis: { title: 'Trading Date', type: 'date', gridcolor: '#d9d0c6', zeroline: false, tickformat: '%b %-d' },
    yaxis: { title: 'Market Direction', tickmode: 'array', tickvals: [-1, 0, 1], ticktext: ['DOWN', 'NEUTRAL', 'UP'], range: [-1.35, 1.35], gridcolor: '#d9d0c6', zeroline: false },
  }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });

  if (message) message.textContent = `${correct} correct and ${incorrect} incorrect prediction${incorrect === 1 ? '' : 's'} across ${validRows.length} January trading days.`;
}

function renderAll() {
  renderStatus();
  renderMetrics();
  renderPrediction();
  renderRisk();
  renderTradeDecision();
  renderChart();
  renderNews();
  renderJanuaryPerformance();
}

async function loadData() {
  const button = document.getElementById('load-data');
  const status = document.getElementById('global-status');
  if (button) button.disabled = true;
  if (button) button.querySelector('.button-label').textContent = 'Refreshing...';
  if (status) {
    status.className = 'global-status';
    status.textContent = 'Updating market data...';
  }

  const requests = [
    fetchOptional('health', API.health), fetchOptional('latest', API.latest), fetchOptional('history', API.history),
    fetchOptional('prediction', API.prediction), fetchOptional('risk', API.risk), fetchOptional('news', API.news),
  ];
  const januaryRequest = fetch(JANUARY_XGBOOST_DATA).then(response => {
    if (!response.ok) throw new Error(`January comparison data returned HTTP ${response.status}`);
    return response.json();
  }).then(data => { state.results.januaryPerformance = data; delete state.errors.januaryPerformance; }).catch(error => {
    state.results.januaryPerformance = null;
    state.errors.januaryPerformance = error.message;
  });
  await Promise.allSettled([...requests, januaryRequest]);

  state.lastUpdated = new Date();
  renderAll();

  const failed = Object.keys(state.errors);
  if (status) {
    status.textContent = failed.length ? `Unable to update market data. Showing last available data.` : 'Market data refreshed successfully.';
    status.className = failed.length ? 'global-status error' : 'global-status';
  }

  if (button) {
    button.disabled = false;
    button.querySelector('.button-label').textContent = 'Refresh data';
  }
}

document.getElementById('load-data').addEventListener('click', loadData);
document.querySelectorAll('.timeframe').forEach(button => button.addEventListener('click', () => selectTimeframe(button.dataset.timeframe)));
const capitalInput = document.getElementById('investment-capital');
const allocationRange = document.getElementById('allocation-percent');
if (capitalInput) {
  capitalInput.value = String(sessionStorage.getItem('hdfc_investment_capital') || '');
  capitalInput.addEventListener('input', () => {
    const value = Number(capitalInput.value || 0);
    sessionStorage.setItem('hdfc_investment_capital', String(value > 0 ? value : 0));
    renderCapitalPlan();
  });
}
document.getElementById('calculate-plan')?.addEventListener('click', calculateTradePlan);
if (allocationRange) {
  allocationRange.value = String(sessionStorage.getItem('hdfc_allocation_percent') || 40);
  allocationRange.addEventListener('input', () => {
    sessionStorage.setItem('hdfc_allocation_percent', allocationRange.value);
    renderCapitalPlan();
  });
}

state.lastUpdated = new Date();
loadData();
setInterval(() => {
  if (document.visibilityState === 'visible') {
    loadData();
  }
}, 60000);
