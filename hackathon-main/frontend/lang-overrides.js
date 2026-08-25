/* Frontend-only language polish. No API, model or backend behavior is changed. */
(() => {
  const polish = {
    'nav.dashboard': 'Panel główny',
    'nav.diagnostics': 'Diagnostyka',
    'nav.inspection': 'Przegląd wyników',
    'nav.insight': 'Informacje o modelu',
    'sidebar.cpu': 'Predykcja na CPU',
    'sidebar.model': 'SVC + detekcja anomalii',
    'header.eyebrow': 'DIAGNOSTYKA PRZEMYSŁOWA',
    'header.title': 'Konsola diagnostyczna silnika',
    'status.checking': 'Sprawdzanie połączenia z API',
    'hero.eyebrow': 'ANALIZA WIDMA AKUSTYCZNEGO',
    'hero.title': 'Wykryj usterkę wtryskiwacza i od razu poznaj jej przyczynę.',
    'hero.text': 'Dla każdego wiersza system podaje diagnozę, sygnał anomalii oraz krótkie uzasadnienie: nietypowe pasmo widma, porównanie z pozostałymi cylindrami i cechy, które miały największy wpływ na decyzję modelu.',
    'hero.modelLabel': 'MODEL',
    'hero.loadingFeatures': 'wczytywanie informacji o cechach…',
    'hero.anomalyLabel': 'ANOMALIE',
    'hero.anomalyText': 'model anomalii uczony wyłącznie na train.csv',
    'measurement.eyebrow': '01 / POMIAR',
    'measurement.title': 'Wczytaj pomiar',
    'measurement.drop': 'Przeciągnij tutaj plik z pomiarem',
    'measurement.browse': 'lub wybierz plik CSV z dysku',
    'measurement.run': 'URUCHOM DIAGNOSTYKĘ',
    'common.remove': 'Usuń plik',
    'loading.title': 'ANALIZOWANIE POMIARU',
    'loading.text': 'Cechy · model · detekcja anomalii · wyjaśnienie wyniku',
    'error.title': 'Nie udało się przeprowadzić analizy',
    'error.default': 'Nie można było przetworzyć tego pomiaru.',
    'result.eyebrow': '02 / WYNIK',
    'result.title': 'Podsumowanie diagnozy',
    'result.waiting': 'OCZEKIWANIE',
    'result.empty': 'Uruchom diagnostykę, aby zobaczyć wyniki.',
    'result.rowsAnalyzed': 'PRZEANALIZOWANE WIERSZE',
    'result.meanConfidence': 'ŚREDNIA PEWNOŚĆ MODELU',
    'result.faultClasses': 'Wykryte klasy usterek',
    'result.mostCommon': 'Najczęstsza diagnoza',
    'result.highAnomaly': 'Silna anomalia',
    'result.analysis': 'Czas analizy',
    'inspection.eyebrow': '03 / PRZEGLĄD',
    'inspection.title': 'Wyniki predykcji',
    'charts.fault': 'Rozkład diagnoz',
    'charts.severity': 'Rozkład stopnia nasilenia',
    'charts.anomaly': 'Rozkład poziomu anomalii',
    'table.filter': 'Filtruj po silniku, cylindrze lub diagnozie…',
    'table.allConfidence': 'Dowolna pewność',
    'table.export': 'Eksportuj JSON',
    'table.engine': 'Silnik',
    'table.cylinder': 'Cyl.',
    'table.diagnosis': 'Diagnoza',
    'table.severity': 'Nasilenie',
    'table.confidence': 'Pewność',
    'table.anomaly': 'Anomalia',
    'table.empty': 'Brak wyników spełniających wybrane kryteria.',
    'why.eyebrow': '04 / UZASADNIENIE',
    'why.title': 'Uzasadnienie diagnozy',
    'why.select': 'Wybierz wynik',
    'why.verdict': 'DIAGNOZA',
    'why.click': 'Wybierz wiersz z wynikami, aby zobaczyć szczegóły i uzasadnienie decyzji modelu.',
    'why.band': 'Nietypowe pasmo',
    'why.deviation': 'Odchylenie od pozostałych cylindrów',
    'why.anomaly': 'Wynik anomalii',
    'why.confidence': 'Pewność',
    'why.features': 'CECHY O NAJWIĘKSZYM WPŁYWIE',
    'insight.eyebrow': '05 / INFORMACJE O MODELU',
    'insight.title': 'Znaczenie cech',
    'insight.meta': 'cechy wybrane na podstawie ich znaczenia',
    'footer.text': 'Wsparcie diagnostyczne — najważniejsze ustalenia należy zawsze zweryfikować zgodnie z procedurą inżynierską.',
    'status.modelOnline': 'MODEL GOTOWY',
    'status.apiStartup': 'API DZIAŁA / URUCHAMIANIE MODELU',
    'status.offline': 'API NIEDOSTĘPNE',
    'status.modelUnavailable': 'model niedostępny',
    'status.backendInsight': 'Uruchom backend, aby wczytać informacje o modelu',
    'status.lowConfidence': 'wierszy o niskiej pewności',
    'status.predictions': 'predykcji',
    'status.rows': 'WIERSZY',
    'status.error': 'BŁĄD',
    'status.waiting': 'OCZEKIWANIE',
    'status.analysisMs': 'ms',
    'error.csv': 'Obsługiwane są wyłącznie pliki CSV.',
    'error.empty': 'Wybrany plik CSV jest pusty.',
    'error.size': 'Wybrany plik przekracza limit API wynoszący 20 MB.',
    'error.timeout': 'Przekroczono limit czasu odpowiedzi API. Sprawdź, czy backend jest uruchomiony.',
    'error.batch': 'Backend nie zwrócił wyników dla przesłanej partii.',
    'theme.light': 'Jasny',
    'theme.dark': 'Ciemny',
    'theme.switchLight': 'Przełącz na jasny motyw',
    'theme.switchDark': 'Przełącz na ciemny motyw',
    'charts.spectrumEyebrow': 'WIDMO',
    'charts.spectrumTitle': 'Widmo wybranego cylindra'
  };

  if (typeof I18N !== 'undefined' && I18N.pl) Object.assign(I18N.pl, polish);

  const setFlag = () => {
    const flag = document.getElementById('languageFlag');
    if (flag) flag.textContent = currentLang === 'pl' ? '🇵🇱' : '🇬🇧';
    document.querySelectorAll('#languageMenu [data-lang]').forEach(button => {
      const lang = button.dataset.lang;
      const label = button.querySelector('span');
      if (!label) return;
      button.firstChild.textContent = lang === 'pl' ? '🇵🇱 ' : '🇬🇧 ';
    });
  };

  if (typeof applyLanguage === 'function') {
    const originalApplyLanguage = applyLanguage;
    window.applyLanguage = function(lang) {
      originalApplyLanguage(lang);
      setFlag();
    };
  }

  const polishFlags = () => {
    const menu = document.getElementById('languageMenu');
    if (!menu) return;
    menu.querySelector('[data-lang="pl"]')?.replaceChildren(document.createTextNode('🇵🇱 '), Object.assign(document.createElement('span'), {textContent:'Polski'}));
    menu.querySelector('[data-lang="en"]')?.replaceChildren(document.createTextNode('🇬🇧 '), Object.assign(document.createElement('span'), {textContent:'English'}));
    setFlag();
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', polishFlags, {once:true});
  else polishFlags();
})();
