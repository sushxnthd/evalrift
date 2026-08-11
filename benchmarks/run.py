import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from evalrift.core import MCQ, robustness
items=[MCQ(f'q{i}',('correct','b','c','d'),0) for i in range(200)]
result=robustness(lambda x:0,items,50)
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2));print(result)
