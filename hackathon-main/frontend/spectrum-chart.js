/* Spectrum visualization for cylinder diagnostics */
(() => {
  'use strict';
  
  const $ = id => document.getElementById(id);
  let currentRows = [];
  let selectedRow = null;
  
  function drawSpectrum(row, canvas) {
    if (!row || !canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 30, right: 30, bottom: 40, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Extract spectrum data (mV_0 through mV_20)
    const spectrumData = [];
    for (let i = 0; i <= 20; i++) {
      const key = `mV_${i}`;
      spectrumData.push(parseFloat(row[key]) || 0);
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Get computed styles for theming
    const styles = getComputedStyle(document.documentElement);
    const bgColor = styles.getPropertyValue('--bg').trim() || '#0b0f12';
    const panelColor = styles.getPropertyValue('--panel').trim() || '#11171b';
    const lineColor = styles.getPropertyValue('--line').trim() || '#273137';
    const textColor = styles.getPropertyValue('--text').trim() || '#edf2f3';
    const mutedColor = styles.getPropertyValue('--muted').trim() || '#8b999e';
    const accentColor = styles.getPropertyValue('--accent').trim() || '#b8e66e';
    const dangerColor = styles.getPropertyValue('--danger').trim() || '#ff6b6b';
    
    // Background
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);
    
    // Find min/max for scaling
    const minVal = Math.min(...spectrumData) * 0.95;
    const maxVal = Math.max(...spectrumData) * 1.05;
    const range = maxVal - minVal || 1;
    
    // Draw grid lines
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1;
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = mutedColor;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = padding.top + (chartHeight / yTicks) * i;
      const val = maxVal - (range / yTicks) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(1), padding.left - 8, y);
    }
    
    // Draw X-axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xStep = chartWidth / 20;
    for (let i = 0; i <= 20; i += 5) {
      const x = padding.left + xStep * i;
      ctx.fillText(`mV_${i}`, x, height - padding.bottom + 8);
    }
    
    // Calculate points
    const points = spectrumData.map((val, i) => ({
      x: padding.left + (xStep * i),
      y: padding.top + chartHeight - ((val - minVal) / range) * chartHeight
    }));
    
    // Draw filled area under curve
    ctx.beginPath();
    ctx.moveTo(points[0].x, padding.top + chartHeight);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, padding.top + chartHeight);
    ctx.closePath();
    ctx.fillStyle = `${accentColor}22`;
    ctx.fill();
    
    // Draw line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw points
    points.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = panelColor;
      ctx.fill();
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 2;
      ctx.stroke();
    });
    
    // Highlight anomalous band if available
    if (row.explanation && row.explanation.anomalous_band) {
      const bandMatch = row.explanation.anomalous_band.match(/mV_(\d+)/);
      if (bandMatch) {
        const bandIdx = parseInt(bandMatch[1]);
        const bandX = padding.left + xStep * bandIdx;
        ctx.beginPath();
        ctx.moveTo(bandX, padding.top);
        ctx.lineTo(bandX, padding.top + chartHeight);
        ctx.strokeStyle = dangerColor;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    
    // Title
    ctx.fillStyle = textColor;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`Engine: ${row.engine_id} | Cylinder: ${row.cylinder}`, padding.left, 8);
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = mutedColor;
    ctx.fillText(`Diagnosis: ${row.label} | Severity: ${row.severity}`, padding.left, 20);
  }
  
  window.renderSpectrumChart = function(rows) {
    currentRows = rows || [];
    const canvas = $('spectrumChart');
    const target = $('spectrumTarget');
    
    if (!canvas || !currentRows.length) {
      if (target) target.textContent = '—';
      return;
    }
    
    // Select first row by default or previously selected
    if (!selectedRow || !currentRows.includes(selectedRow)) {
      selectedRow = currentRows[0];
    }
    
    if (target) {
      target.textContent = `${selectedRow.engine_id} / cylinder ${selectedRow.cylinder}`;
    }
    
    // Need to fetch full row data with spectrum values
    // For now, show placeholder - actual implementation needs spectrum data from API
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#8b999e';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Spectrum visualization requires raw measurement data', canvas.width / 2, canvas.height / 2);
  };
  
  window.selectSpectrumRow = function(row) {
    selectedRow = row;
    if (currentRows.includes(row)) {
      window.renderSpectrumChart(currentRows);
    }
  };
})();
