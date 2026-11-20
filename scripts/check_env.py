import importlib
import sys

packages = [
    ("python", sys.version.split()[0]),
    ("numpy", None),
    ("pandas", None),
    ("sklearn", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("skimage", None),
    ("PIL", None),
    ("tqdm", None),
    ("requests", None),
    ("astropy", None),
    ("sunpy", None),
    ("astroquery", None),
]

print("Environment check:\n------------------")
print(f"python: {packages[0][1]}")

for name, _ in packages[1:]:
    try:
        m = importlib.import_module(name)
        version = getattr(m, "__version__", "unknown")
        print(f"{name}: {version}")
    except Exception as e:
        print(f"{name}: MISSING ({e.__class__.__name__}: {e})")
