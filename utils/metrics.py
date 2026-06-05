from pathlib import Path

path = Path("models\ensemble")
for file in path.glob('*.py'):
    print(file)


