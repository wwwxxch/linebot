
### Development
Virtual Environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Start server
```
python src/app.py
```
Deactivate
```
deactivate
```

### Production
Build
```
pip install -r requirements.txt
```
Start server
```
gunicorn --bind 0.0.0.0:3000 src.app:app
```