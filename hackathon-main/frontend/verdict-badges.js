(() => {
  const POSITIVE = new Set(['ok']);
  const NEUTRAL = new Set(['unknown']);
  const NEGATIVE = new Set(['zakoksowany', 'lejacy', 'pompa', 'iglica']);
  function badgeClass(label) { const value=String(label||'').trim().toLowerCase(); if(POSITIVE.has(value))return 'verdict-badge--positive'; if(NEUTRAL.has(value))return 'verdict-badge--neutral'; if(NEGATIVE.has(value))return 'verdict-badge--negative'; return ''; }
  function decorate(){document.querySelectorAll('#resultsBody td:nth-child(4), #explainTitle').forEach(cell=>{if(cell.dataset.verdictBadge==='1')return;const cls=badgeClass(cell.textContent);if(!cls)return;const text=cell.textContent.trim();cell.textContent='';const badge=document.createElement('span');badge.className=`verdict-badge ${cls}`;badge.textContent=text;cell.appendChild(badge);cell.dataset.verdictBadge='1';});}
  const originalRenderTable=window.renderTable;
  if(typeof originalRenderTable==='function'){window.renderTable=function(...args){const result=originalRenderTable.apply(this,args);decorate();return result;};}
  const observer=new MutationObserver(decorate);const body=document.getElementById('resultsBody');if(body)observer.observe(body,{childList:true,subtree:true});const explanation=document.getElementById('explainTitle');if(explanation)observer.observe(explanation,{childList:true,characterData:true,subtree:true});decorate();
  const css=document.createElement('link');css.rel='stylesheet';css.href='./inspection-tools.css';document.head.appendChild(css);
  const script=document.createElement('script');script.src='./inspection-tools.js';script.defer=true;document.body.appendChild(script);
})();
