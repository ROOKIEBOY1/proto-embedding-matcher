"""Audit saved experimental evidence and recompute real-model cosine scores."""
import csv
import json
import math
from pathlib import Path
import proto_embedding_p0 as p


def main():
    """Validate counts, scores, labels, calibration hygiene and independent replay."""
    with (p.ROOT/'results.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    data=json.loads(p.DATA.read_text())
    assert len(rows)==len(data)*24
    assert len({(r['id'],r['strategy'],r['retrieval'],r['method']) for r in rows})==len(rows)
    for r in rows:
        for n in [1,2,3]:
            assert math.isfinite(float(r[f'top{n}_score']))
            assert -1.00001<=float(r[f'top{n}_score'])<=1.00001
        assert float(r['top1_score'])>=float(r['top2_score'])>=float(r['top3_score'])
        assert abs(float(r['margin'])-(float(r['top1_score'])-float(r['top2_score'])))<2e-7
        if r['retrieval']=='hierarchical':
            assert r['top1_enum']==r['top2_enum']==r['top3_enum']==r['field_predicted']
        known=bool(r['expected'])
        rejected=r['rejected']=='True'
        correct=(not rejected and r['predicted']==r['expected'] and r['predicted_enum']==r['expected_enum']) if known else rejected
        assert (r['correct']=='True')==correct
        if r['method']=='forced':
            assert not rejected
    thresholds=json.loads((p.OUT/'thresholds.json').read_text())
    for key,info in thresholds.items():
        strategy,retrieval=key.split('/')
        dev=[r for r in rows if r['split']=='dev' and r['strategy']==strategy and r['retrieval']==retrieval and r['method']=='forced']
        assert len(dev)==112
        actual=p.calibrate([(float(r['top1_score']),bool(r['expected'])) for r in dev])
        assert abs(actual-info['threshold'])<1e-10
    summary=json.loads((p.OUT/'summary.json').read_text())
    for s in summary:
        group=[r for r in rows if r['split']=='test' and r['strategy']==s['strategy'] and r['retrieval']==s['retrieval'] and r['method']==s['method']]
        known=[r for r in group if r['expected']]
        unknown=[r for r in group if not r['expected']]
        assert len(known)==184 and len(unknown)==40
        assert abs(sum(r['correct']=='True' for r in known)/184-s['top1_accuracy'])<1e-10
        assert abs(sum(r['rejected']=='False' for r in unknown)/40-s['false_acceptance_rate'])<1e-10
    candidates=p.parse_proto(p.PROTO.read_text())
    engine=p.Engine(p.MODEL)
    replay=[]
    for value in ['48 FPS','144 FPS']:
        # 48 FPS is the canonical positive under field Video Frame Rate;
        # 144 FPS is the separately labeled unsupported query.
        field='Video Frame Rate' if value=='48 FPS' else 'Frame Rate'
        q=engine.encode(field+': '+value)
        f=engine.encode(field)
        for strategy in 'ABC':
            for hierarchical in [False,True]:
                order,scores,*_=engine.rank(candidates,strategy,hierarchical,q,f)
                saved=next(r for r in rows if r['field']==field and r['value']==value and r['strategy']==strategy and r['retrieval']==('hierarchical' if hierarchical else 'global') and r['method']=='forced')
                for n,i in enumerate(order[:3],1):
                    assert candidates[i].name==saved[f'top{n}_name']
                    assert abs(float(scores[i])-float(saved[f'top{n}_score']))<2e-5
                replay.append(dict(field=field,value=value,strategy=strategy,hierarchical=hierarchical,top1=candidates[order[0]].name,score=float(scores[order[0]])))
    result=dict(status='PASS',csv_rows=len(rows),configurations=len(summary),
                dev_threshold_recalculations=len(thresholds),real_model_score_replays=replay)
    p.dump_json(p.OUT/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
