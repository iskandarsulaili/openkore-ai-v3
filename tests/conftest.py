"""pytest configuration — adds AI_sidecar to Python path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "AI_sidecar"))
