Environement wiederherstellen:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

In Python Projekt setzen:
Zahnrad -> Settings -> Python -> Interpreter -> AddInterpreter -> Select Existing

Update der requirements:
pip freeze > requirements.txt
