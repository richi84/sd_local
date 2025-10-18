Environement wiederherstellen:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

In Python Projekt setzen:
Zahnrad -> Settings -> Python -> Interpreter -> AddInterpreter -> Select Existing

Update der requirements:
pip freeze > requirements.txt

### Hinweis zu Detektoren

Das Projekt setzt ausschließlich auf den MediaPipe-basierten Handdetektor. Über
`detectors.build_available_detectors` wird lediglich `MediaPipeHandsDetector`
initialisiert; zusätzliche Fallbacks oder alternative Modelle wurden entfernt,
dum den Ablauf so schlank wie möglich zu halten. Schlägt die Initialisierung
fehl (z. B. weil MediaPipe nicht installiert ist), wird dies über den
zurückgegebenen `DetectorStatus` gemeldet und der Refinement-Schritt übersprungen.
