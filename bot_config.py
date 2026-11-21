import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DEFAULTS = {'adb': r'C:\Users\LENOVO\AppData\Local\Android\Sdk\platform-tools\adb.exe', 'port': 5555, 'device': None, 'debug_dir': 'debug', 'timeout': 15, 'match_threshold': 0.88, 'stable_threshold': 1.5, 'camera_threshold': 3.0, 'templates_dir': 'templates', 'rois': {}}
path = ROOT / 'bot_config.json'
CONFIG = dict(DEFAULTS)
if path.exists():
    CONFIG.update(json.loads(path.read_text(encoding='utf-8-sig')))
def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
