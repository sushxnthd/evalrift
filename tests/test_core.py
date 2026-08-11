from evalrift.core import MCQ,reorder,robustness

def test_reorder_preserves_answer_text():
    x=MCQ('q',('a','b','c'),1); y=reorder(x,4); assert y.choices[y.answer]=='b'

def test_sensitive_baseline_drops():
    items=[MCQ(str(i),('ok','x','y','z'),0) for i in range(30)]
    assert robustness(lambda x:0,items,10)['max_drop']>0


def test_relabel_preserves_correct_text():
    from evalrift.core import relabel
    x=MCQ('q',('x','correct','z'),1); r=relabel(x)
    assert r['choices'][r['answer']]=='correct'

def test_ranking_instability_bounded():
    from evalrift.core import ranking_instability
    items=[MCQ(str(i),('ok','x','y','z'),0) for i in range(20)]
    v=ranking_instability({'first':lambda x:0,'second':lambda x:1},items,5)
    assert 0<=v<=1
