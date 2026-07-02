import sys
import traceback
from features import extract_features

path = sys.argv[1]
try:
    vec = extract_features(path)
    print("SUCCESS:", vec)
except Exception:
    print("FAILED with full traceback:")
    traceback.print_exc()