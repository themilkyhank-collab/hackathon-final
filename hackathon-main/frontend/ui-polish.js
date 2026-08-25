(() => {
  'use strict';

  const injectUiStyles = () => {
    if (document.getElementById('aesteel-ui-polish-styles')) return;
    const style = document.createElement('style');
    style.id = 'aesteel-ui-polish-styles';
    style.textContent = `
      /* Dashboard-like focus for the section currently in view. */
      .section-focus {
        position: relative;
        border-color: color-mix(in srgb, var(--accent) 42%, var(--line));
        box-shadow: 0 18px 45px rgba(0,0,0,.16), inset 0 2px 0 var(--accent);
        transform: translateY(-2px);
      }
      .section-focus > .panel-head {
        padding-bottom: 14px;
        border-bottom: 1px solid color-mix(in srgb, var(--accent) 18%, var(--line));
      }
      .section-focus .eyebrow { color: var(--accent); }
      .nav-item.active { font-weight: 800; }

      /* Never truncate chart labels. Give the label column room and allow wrapping. */
      .chart-row { grid-template-columns: minmax(120px, 150px) minmax(70px, 1fr) 34px; }
      .chart-row > span {
        overflow: visible;
        text-overflow: clip;
        white-space: normal;
        overflow-wrap: anywhere;
        line-height: 1.25;
      }
      .chart-card { min-width: 0; }

      /* Export toolbar. */
      .export-tools {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-left: auto;
        flex-shrink: 0;
      }
      .export-label {
        color: var(--muted);
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-right: 3px;
      }
      .export-btn {
        border: 1px solid var(--accent);
        background: var(--accent);
        color: #fff;
        min-width: 92px;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .04em;
        cursor: pointer;
        transition: background .18s ease, color .18s ease, transform .18s ease, box-shadow .18s ease;
      }
      .export-btn:hover {
        background: #fff;
        color: var(--accent);
        box-shadow: 0 7px 18px rgba(0,0,0,.12);
        transform: translateY(-1px);
      }
      .export-btn:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
      }
      @media (max-width: 760px) {
        .table-tools { flex-wrap: wrap; }
        .export-tools { width: 100%; margin-left: 0; flex-wrap: wrap; }
      }
      @media print {
        .sidebar, .topbar-controls, .upload-panel, .result-panel, .table-tools, footer { display: none !important; }
        .app-shell { display: block; }
        .main { max-width: none; padding: 20px; }
        #inspection { display: block !important; box-shadow: none; border: 0; }
        .panel { box-shadow: none !important; transform: none !important; }
      }
    `;
    document.head.appendChild(style);
  };

  const replaceModelText = () => {
    const replacements = new Map([
      ['XGBoost + anomaly detection', 'RBF SVC + anomaly detection'],
      ['Features · XGBoost · anomaly detection · explanations', 'Features · RBF SVC · anomaly detection'],
      ['Cechy · XGBoost · detekcja anomalii · wyjaśnienia', 'Cechy · RBF SVC · detekcja anomalii'],
      ['XGBoost', 'RBF SVC'],
      ['Ważność cech', 'Wyniki diagnostyki'],
      ['Feature importance', 'Diagnostic results'],
      ['importance-driven selection', 'prediction and anomaly analysis'],
      ['selekcja według ważności', 'analiza predykcji i anomalii']
    ]);
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(node => {
      let value = node.nodeValue;
      replacements.forEach((to, from) => { value = value.split(from).join(to); });
      if (value !== node.nodeValue) node.nodeValue = value;
    });
  };

  const removeFeatureImportance = () => {
    const section = document.getElementById('features');
    if (section) section.remove();
    document.querySelectorAll('.nav-item').forEach(item => {
      if (item.getAttribute('href') === '#features') item.remove();
    });
  };

  const csvEscape = value => {
    const text = String(value ?? '');
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };

  const exportVisibleTableCsv = () => {
    const table = document.querySelector('#resultsBody');
    if (!table || !table.rows.length) return;
    const headers = [...document.querySelectorAll('table thead th')].map(th => th.textContent.trim());
    const rows = [...table.rows].map(row => [...row.cells].map(cell => cell.textContent.trim()));
    const csv = [headers, ...rows].map(row => row.map(csvEscape).join(',')).join('\r\n');
    const blob = new Blob([`\ufeff${csv}`], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aesteel-diagnostics-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const exportPdf = () => {
    // Uses the browser's native PDF printer, avoiding a heavy client-side PDF library.
    const previousTitle = document.title;
    document.title = `Aesteel diagnostics ${new Date().toISOString().slice(0, 10)}`;
    window.print();
    setTimeout(() => { document.title = previousTitle; }, 500);
  };

  const addExportControls = () => {
    const jsonButton = document.getElementById('exportButton');
    if (!jsonButton) return;
    let group = document.getElementById('exportTools');
    if (!group) {
      group = document.createElement('div');
      group.id = 'exportTools';
      group.className = 'export-tools';
      const label = document.createElement('span');
      label.className = 'export-label';
      label.textContent = 'EXPORT';
      group.appendChild(label);
      jsonButton.parentNode.insertBefore(group, jsonButton);
      group.appendChild(jsonButton);
    }
    jsonButton.classList.add('export-btn');
    jsonButton.textContent = 'JSON';

    let csvButton = document.getElementById('exportCsvButton');
    if (!csvButton) {
      csvButton = document.createElement('button');
      csvButton.id = 'exportCsvButton';
      csvButton.type = 'button';
      csvButton.textContent = 'CSV';
      csvButton.addEventListener('click', exportVisibleTableCsv);
      group.appendChild(csvButton);
    } else if (csvButton.parentElement !== group) {
      group.appendChild(csvButton);
    }
    csvButton.classList.add('export-btn');
    csvButton.textContent = 'CSV';

    let pdfButton = document.getElementById('exportPdfButton');
    if (!pdfButton) {
      pdfButton = document.createElement('button');
      pdfButton.id = 'exportPdfButton';
      pdfButton.type = 'button';
      pdfButton.textContent = 'PDF';
      pdfButton.addEventListener('click', exportPdf);
      group.appendChild(pdfButton);
    }
    pdfButton.classList.add('export-btn');
  };

  const initSectionFocus = () => {
    const ids = ['diagnostics', 'measurement', 'inspection', 'explanation'];
    const sections = ids.map(id => document.getElementById(id)).filter(Boolean);
    const setFocus = id => sections.forEach(section => section.classList.toggle('section-focus', section.id === id));
    if (!('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(e => e.isIntersecting).sort((a,b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setFocus(visible.target.id);
    }, { rootMargin: '-18% 0px -58% 0px', threshold: [0.1, 0.25, 0.5] });
    sections.forEach(section => observer.observe(section));
  };

  const apply = () => {
    injectUiStyles();
    replaceModelText();
    removeFeatureImportance();
    addExportControls();
    initSectionFocus();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', apply, { once: true });
  } else {
    apply();
  }

  document.addEventListener('click', event => {
    if (event.target.closest('#languageMenu [data-lang]')) setTimeout(apply, 0);
  });
})();
