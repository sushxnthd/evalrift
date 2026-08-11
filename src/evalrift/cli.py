import argparse, json
from pathlib import Path
from .core import MCQ, robustness

def main():
    p=argparse.ArgumentParser(description='Audit answer-order sensitivity in an MCQ benchmark.')
    p.add_argument('dataset'); p.add_argument('--rounds',type=int,default=20);a=p.parse_args()
    rows=json.loads(Path(a.dataset).read_text()); items=[MCQ(r['question'],tuple(r['choices']),r['answer']) for r in rows]
    model=lambda x:0  # explicit baseline: always choose first presented option
    print(json.dumps(robustness(model,items,a.rounds),indent=2))
if __name__=='__main__':main()
