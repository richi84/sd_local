Environement wiederherstellen:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

In Python Projekt setzen:
Zahnrad -> Settings -> Python -> Interpreter -> AddInterpreter -> Select Existing

Update der requirements:
pip freeze > requirements.txt

### Hinweis zu Detektoren

Falls optionale Bibliotheken wie MediaPipe oder die Ultralytics-YOLO-Implementierung
nicht installiert sind, greift automatisch der neue `SimpleSkinDetector`. Dieser
erstellt grobe Bounding-Boxes auf Basis eines einfachen Hautton-Filters, so dass
`detections.json` nicht mehr leer bleibt und der ADetailer-Refine trotzdem eine
Maske generieren kann. Für präzisere Ergebnisse können weiterhin spezialisierte
Hand-/Gesichtsdetektoren ergänzt werden. Zusätzlich steht ein OpenPose-Wrapper
(`OpenposeDetector`) zur Verfügung, der über die ControlNet-Preprozessoren
(`controlnet-aux`) Körper-, Hand- und Gesicht-Keypoints erkennt und daraus
exakte Maskenbereiche ableitet.
