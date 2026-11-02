"""Offline BGE camera-parameter benchmark; see README.md and REPORT.md.

All inference scores originate in SentenceTransformer.encode. Fixtures are
synthetic and frozen before inference. Core logic deliberately lives here.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import time

ROOT = Path(__file__).resolve().parent
PROTO = ROOT / 'data/camera.proto'
DATA = ROOT / 'data/dataset.json'
MODEL = ROOT / 'models/bge-small-en-v1.5'
OUT = ROOT / 'results'
COMPOUNDS = {'slowmotion':'Slow Motion', 'videonormal':'Normal Video',
             'timelapse':'Time Lapse', 'daylight':'Daylight', 'mjpeg':'MJPEG'}


@dataclass(frozen=True)
class Candidate:
    """An enum member identity includes its enum type, name AND wire number."""
    enum: str
    name: str
    number: int


def parse_proto(text: str) -> list[Candidate]:
    """Parse enum declarations from a small Proto subset, ignoring comments.

    Accept signed IDs and enum member options. Does not resolve imports, package
    scopes, nested same-name types, or preprocessor directives. Reject duplicate
    enum type names to avoid silently conflating identities.
    """
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    result, seen = [], set()
    for enum, body in re.findall(r'\benum\s+(\w+)\s*\{([^{}]*)\}', text):
        if enum in seen:
            raise ValueError('Ambiguous duplicate enum type: ' + enum)
        seen.add(enum)
        for name, number in re.findall(r'\b(\w+)\s*=\s*([+-]?\d+)\s*(?:\[[^\]]*\])?\s*;', body):
            result.append(Candidate(enum, name, int(number)))
    if not result:
        raise ValueError('No enum members found')
    return result


def words(text: str) -> str:
    """Split underscores, CamelCase and digit-to-unit boundaries; no synonyms."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)
    return ' '.join(text.replace('_', ' ').split())


def field_text(enum: str, strategy: str) -> str:
    """Return enum type encoding under the same A/B/C policy as candidates."""
    if strategy == 'A':
        return enum
    text = words(enum)
    return re.sub(r'\s+Type$', '', text) if strategy == 'C' else text


def value_text(item: Candidate, strategy: str) -> str:
    """Remove common field prefix for B/C; apply only documented C rules."""
    prefix = re.sub(r'(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])', '_', item.enum).upper()
    suffix = item.name
    for pref in [prefix + '_', prefix.removesuffix('_TYPE') + '_']:
        if suffix.startswith(pref):
            suffix = suffix[len(pref):].lstrip('_')
            break
    if strategy == 'C':
        if suffix.lower() in COMPOUNDS:
            return COMPOUNDS[suffix.lower()]
        suffix = re.sub(r'^MINUS_', '-', suffix)
        suffix = re.sub(r'^PLUS_', '+', suffix)
        suffix = re.sub(r'^ZERO_', '', suffix)
    text = words(suffix)
    text = ' '.join(w if any(c.isdigit() for c in w) or w in {'FPS','EV','ISO','MJPEG'}
                    else w.capitalize() for w in text.split())
    return text


def describe(item: Candidate, strategy: str) -> str:
    """Encode raw names (A), mechanical splitting (B), or explicit rules (C)."""
    if strategy == 'A':
        return item.name
    return field_text(item.enum, strategy) + ': ' + value_text(item, strategy)


def numbers(text: str) -> tuple[Decimal, ...]:
    """Extract physical numeric tokens, preserving sign and decimal equality.

    H.264/H.265 are normalized as codec names. Enum wire IDs never enter here.
    This is a numeric equality check, not a general units/conversion parser.
    """
    text = re.sub(r'\bH\.(26[45])\b', r'H\1', text, flags=re.I)
    return tuple(Decimal(x) for x in re.findall(r'[+-]?\d+(?:\.\d+)?', text))


def numeric_candidates(value: str, candidates: list[Candidate]) -> list[int] | None:
    """Return all exact numeric matches within a predicted enum, not just top-3.

    None means no query number (guard cannot establish support); [] means a
    numeric query has no exact candidate. Indices refer to the supplied list.
    """
    query = numbers(value)
    if not query:
        return None
    return [i for i, c in enumerate(candidates) if numbers(value_text(c, 'C')) == query]


def calibrate(scores: list[tuple[float, bool]]) -> float:
    """Choose dev threshold maximizing balanced support acceptance accuracy.

    Labels indicate supported vs unsupported, not whether top-1 is correct.
    Break ties toward the higher (more conservative) threshold. No test rows.
    """
    vals = sorted(set(s for s, _ in scores))
    candidates = [vals[0] - 1e-7] + [(a+b)/2 for a,b in zip(vals, vals[1:])] + [vals[-1]+1e-7]
    positive = [s for s,k in scores if k]
    negative = [s for s,k in scores if not k]
    if not positive or not negative:
        raise ValueError('Calibration needs known and unknown dev examples')
    return max(candidates, key=lambda t: (
        (sum(s >= t for s in positive)/len(positive) + sum(s < t for s in negative)/len(negative))/2, t))


def dump_json(path: Path, obj):
    """Persist JSON with readable Unicode and stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path: Path, rows: list[dict]):
    """Write all raw inference evidence in UTF-8 BOM CSV for Excel consumers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Engine:
    """One offline model; reusable candidate vectors per scheme and candidate set."""
    def __init__(self, model_path: Path, device: str = 'auto', repeats: int = 3):
        """Load local safetensors, measure RSS/CUDA load deltas, and warm up."""
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HOME'] = str(ROOT/'.cache/huggingface')
        os.environ['HF_HUB_CACHE'] = str(ROOT/'.cache/huggingface/hub')
        (ROOT/'.cache/huggingface/hub').mkdir(parents=True,exist_ok=True)
        import numpy as np
        import psutil
        import torch
        from sentence_transformers import SentenceTransformer
        self.np, self.torch = np, torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.repeats = repeats
        self.cache = {}
        process = psutil.Process()
        if self.device == 'cuda':
            torch.cuda.init()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        before = process.memory_info().rss
        start = time.perf_counter()
        self.model = SentenceTransformer(str(model_path), device=self.device, local_files_only=True,
                                         model_kwargs={'use_safetensors':True})
        self.sync()
        self.load_ms = (time.perf_counter()-start)*1000
        self.model_rss_delta_mb = (process.memory_info().rss-before)/2**20
        self.rss_after_load_mb = process.memory_info().rss/2**20
        start = time.perf_counter()
        for _ in range(3):
            self.encode('Camera Mode: Slow Motion')
        self.warmup_ms = (time.perf_counter()-start)*1000
        self.rss_warm_mb = process.memory_info().rss/2**20
        self.cuda_allocated_mb = torch.cuda.memory_allocated()/2**20 if self.device == 'cuda' else 0
        self.cuda_reserved_mb = torch.cuda.memory_reserved()/2**20 if self.device == 'cuda' else 0

    def sync(self):
        """Synchronize CUDA so timings include actual execution, not launch only."""
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def encode(self, texts):
        """Produce normalized float32 BGE vectors; dot product equals cosine."""
        result = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False, batch_size=32)
        self.sync()
        return result

    def timed_encode(self, text: str):
        """Run sequential encoding repeats; return vector and every measured ms."""
        times = []
        for _ in range(self.repeats):
            start = time.perf_counter()
            vector = self.encode(text)
            times.append((time.perf_counter()-start)*1000)
        return vector, times

    def prepare(self, candidates: list[Candidate], strategy: str):
        """Precompute candidate and enum type embeddings once per configuration."""
        key = (tuple(candidates), strategy)
        if key not in self.cache:
            enums = list(dict.fromkeys(x.enum for x in candidates))
            start = time.perf_counter()
            vectors = self.encode([describe(c, strategy) for c in candidates])
            candidate_ms = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            fields = self.encode([field_text(e, strategy) for e in enums])
            field_ms = (time.perf_counter()-start)*1000
            self.cache[key] = (vectors, fields, enums, candidate_ms, field_ms)
        return self.cache[key]

    def rank(self, candidates, strategy, hierarchical, query_vector, field_vector):
        """Rank using global cosine or predicted enum then within-enum cosine.

        No label, wire number, alias table or numeric guard affects pure ranks.
        Returns ordered full candidate indices, all scores, predicted enum,
        field score and matching time including filtering and sorting.
        """
        vectors, fields, enums, _, _ = self.prepare(candidates, strategy)
        start = time.perf_counter()
        field_scores = fields @ field_vector
        selected = enums[int(field_scores.argmax())]
        scores = vectors @ query_vector
        order = self.np.argsort(-scores, kind='stable').tolist()
        if hierarchical:
            order = [i for i in order if candidates[i].enum == selected]
        else:
            selected = candidates[order[0]].enum
        return order, scores, selected, float(field_scores.max()), (time.perf_counter()-start)*1000


def make_record(sample, candidates, strategy, hierarchical, ranked, threshold, method,
                query_times, field_times):
    """Apply an explicitly named decision policy and preserve raw pure top-3.

    Numeric-only is an extra safety control; combined adds the dev threshold.
    If numeric filtering finds a candidate outside pure top-3 it can select it.
    Scores stay real original cosine scores; rejection uses UNKNOWN.
    """
    order, scores, enum, field_score, match_ms = ranked
    decision_start = time.perf_counter()
    top = candidates[order[0]]
    selected_index = order[0]
    rejected, reason = False, ''
    if method in ('numeric', 'combined'):
        pool = [i for i,c in enumerate(candidates) if c.enum == enum]
        exact = numeric_candidates(sample['value'], [candidates[i] for i in pool])
        if exact == []:
            rejected, reason = True, 'NUMERIC_UNSUPPORTED_IN_PREDICTED_ENUM'
        elif exact is not None:
            allowed = [pool[i] for i in exact]
            selected_index = max(allowed, key=lambda i: float(scores[i]))
    selected_score = float(scores[selected_index])
    if method in ('threshold', 'combined') and not rejected and selected_score < threshold:
        rejected, reason = True, 'BELOW_DEV_THRESHOLD'
    pred = candidates[selected_index]
    known = sample['expected'] is not None
    correct = (not rejected and pred.enum == sample['expected_enum'] and pred.name == sample['expected']) if known else rejected
    decision_ms = (time.perf_counter()-decision_start)*1000
    rec = {**sample, 'strategy':strategy, 'retrieval':'hierarchical' if hierarchical else 'global',
           'method':method, 'predicted': 'UNKNOWN' if rejected else pred.name,
           'predicted_enum':None if rejected else pred.enum,
           'predicted_number':None if rejected else pred.number,
           'score':selected_score, 'threshold':threshold, 'rejected':rejected,
           'reason':reason, 'correct':bool(correct),
           'field_predicted':enum, 'field_score':field_score,
           'field_correct':enum == sample['expected_enum'] if sample['expected_enum'] else None,
           'top3_recall':known and any(candidates[i].enum == sample['expected_enum'] and
                                     candidates[i].name == sample['expected'] for i in order[:3]),
           'margin':float(scores[order[0]]-scores[order[1]]) if len(order)>1 else None,
           'query_encode_ms':statistics.median(query_times),
           'field_encode_ms':statistics.median(field_times) if hierarchical else 0,
           'similarity_ms':match_ms,
           'decision_ms':decision_ms,
           'latency_ms':statistics.median([q+(f if hierarchical else 0)+match_ms+decision_ms
                                          for q,f in zip(query_times,field_times)]),
           'query_repeat_ms':json.dumps(query_times),
           'field_repeat_ms':json.dumps(field_times if hierarchical else []),
           'top1':top.name}
    for rank in range(3):
        index = order[rank] if rank < len(order) else None
        c = candidates[index] if index is not None else None
        rec[f'top{rank+1}_name'] = c.name if c else None
        rec[f'top{rank+1}_enum'] = c.enum if c else None
        rec[f'top{rank+1}_number'] = c.number if c else None
        rec[f'top{rank+1}_score'] = float(scores[index]) if c else None
    return rec


def metrics(rows):
    """Compute known ranking/decision accuracy and unknown rejection separately."""
    import numpy as np
    known = [r for r in rows if r['expected'] is not None]
    unknown = [r for r in rows if r['expected'] is None]
    mean = lambda values: sum(values)/len(values) if values else None
    # Raw top1 is identical across policies; accepted accuracy also charges rejections.
    return dict(n=len(rows),known=len(known),unknown=len(unknown),
                raw_top1_accuracy=mean([r['top1_name']==r['expected'] and r['top1_enum']==r['expected_enum'] for r in known]),
                top1_accuracy=mean([r['correct'] for r in known]),
                top3_recall=mean([r['top3_recall'] for r in known]),
                field_accuracy=mean([r['field_correct'] for r in known]),
                false_acceptance_rate=mean([not r['rejected'] for r in unknown]),
                unknown_rejection_recall=mean([r['rejected'] for r in unknown]),
                known_rejection_rate=mean([r['rejected'] for r in known]),
                overall_decision_accuracy=mean([r['correct'] for r in rows]),
                p50_latency_ms=float(np.percentile([r['latency_ms'] for r in rows],50)),
                p95_latency_ms=float(np.percentile([r['latency_ms'] for r in rows],95)))


def display(row):
    """Print a query, true top-3 cosine evidence and final enum decision."""
    print(f"\nInput: {row['field']}: {row['value']} [{row['strategy']}/{row['retrieval']}/{row['method']}]")
    for rank in range(1,4):
        name, score = row[f'top{rank}_name'], row[f'top{rank}_score']
        print(f'Top-{rank}: {name} = {row[f"top{rank}_number"]}; Score: {score:.6f}' if name else f'Top-{rank}: N/A (enum has fewer than 3 members)')
    if row['margin'] is not None:
        print(f"Top1-Top2 margin: {row['margin']:.6f}")
    print(f"Decision: {row['predicted']} = {row['predicted_number']}; Reason: {row['reason'] or 'accepted'}")
    if row['expected'] is not None or row.get('split') != 'interactive':
        print(f"Expected: {row['expected'] or 'UNKNOWN'}; Result: {'PASS' if row['correct'] else 'FAIL'}")


def environment(engine):
    """Capture actual runtime, artifact hashes, device, memory and versions."""
    import psutil
    packages = ['torch','sentence-transformers','transformers','huggingface-hub','numpy','psutil','safetensors','tokenizers']
    files = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [PROTO, DATA, ROOT/'data/naming_variants.json', MODEL/'model.safetensors']}
    torch = engine.torch
    return dict(python=platform.python_version(),platform=platform.platform(),
                executable=os.sys.executable,device=engine.device,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cuda_build=torch.version.cuda,cpu=platform.processor(),logical_cpus=psutil.cpu_count(),
                ram_total_mb=psutil.virtual_memory().total/2**20,
                ram_available_mb=psutil.virtual_memory().available/2**20,
                model_revision=(MODEL/'REVISION').read_text().strip(),
                versions={p:importlib.metadata.version(p) for p in packages},hashes=files,
                model_load_ms=engine.load_ms,warmup_ms=engine.warmup_ms,
                model_rss_delta_mb=engine.model_rss_delta_mb,rss_after_load_mb=engine.rss_after_load_mb,
                rss_after_warmup_mb=engine.rss_warm_mb,cuda_allocated_mb=engine.cuda_allocated_mb,
                cuda_reserved_mb=engine.cuda_reserved_mb,
                cuda_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if engine.device=='cuda' else 0,
                model_directory_mb=sum(p.stat().st_size for p in MODEL.rglob('*') if p.is_file())/2**20,
                venv_directory_mb=sum(p.stat().st_size for p in (ROOT/'.venv').rglob('*') if p.is_file())/2**20,
                embedding_dimension=engine.model.get_sentence_embedding_dimension(),
                max_sequence_length=engine.model.max_seq_length,
                parameter_count=sum(p.numel() for p in engine.model.parameters()),
                query_instruction='none; symmetric short parameter matching',
                normalization='L2 vectors, float32 numpy dot = cosine',
                repeats=engine.repeats,normalization_rules=COMPOUNDS)


def zero_shot(engine, candidates, samples):
    """Before/after candidate insertion and independently defined naming tests."""
    rows = []
    baseline = [c for c in candidates if c.enum!='ImageTextureType' and not c.name.endswith('1440FPS')]
    cases = [s for s in samples if s['split']=='test' and s['category'] in ('new_enum','new_member')]
    for s in cases:
        q = engine.encode(s['field']+': '+s['value'])
        f = engine.encode(s['field'])
        for stage, pool in [('before',baseline),('after',candidates)]:
            for strategy in 'ABC':
                ranked = engine.rank(pool,strategy,False,q,f)
                rec = make_record(s,pool,strategy,False,ranked,0,'forced',[0],[0])
                rows.append(dict(scenario=s['category'],stage=stage,**rec))
    spec = json.loads((ROOT/'data/naming_variants.json').read_text())
    for i, schema in enumerate(spec['schemas']):
        pool = parse_proto(schema)
        for j, query in enumerate(spec['queries']):
            field,value = query.split(': ',1)
            s=dict(id=f'naming_{i}_{j}',split='test',field=field,value=value,
                   expected=spec['expected'][i],expected_enum=pool[0].enum,category='naming',source='predefined_semantic_contract')
            for strategy in 'ABC':
                ranked=engine.rank(pool,strategy,False,engine.encode(query),engine.encode(field))
                rec=make_record(s,pool,strategy,False,ranked,0,'forced',[0],[0])
                rows.append(dict(scenario='naming',stage=f'schema_{i}',**rec))
    # Normalize keys as naming samples intentionally omit irrelevant metadata.
    keys = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(OUT/'zero_shot.csv',[{k:r.get(k) for k in keys} for r in rows])
    return rows


def evaluate(engine, candidates, samples, args):
    """Run all A/B/C x retrieval x rejection policies on the same frozen split."""
    rows, thresholds, precompute = [], {}, []
    for strategy in 'ABC':
        prep = engine.prepare(candidates,strategy)
        precompute.append(dict(strategy=strategy,candidate_ms=prep[3],field_ms=prep[4],
                               candidate_count=len(candidates),enum_count=len(prep[2]),
                               vector_bytes=prep[0].nbytes+prep[1].nbytes))
    # Measure each sample's query and field encoding sequentially. Embeddings
    # are reused across strategies/methods for fairness, not a per-query cache.
    cached = {}
    for i,s in enumerate(samples):
        q, qt = engine.timed_encode(s['field']+': '+s['value'])
        f, ft = engine.timed_encode(s['field'])
        cached[s['id']] = q,qt,f,ft
        if (i+1)%40==0:
            print(f'Encoded {i+1}/{len(samples)} samples',flush=True)
    for strategy in 'ABC':
        for hierarchical in [False,True]:
            ranks = {s['id']:engine.rank(candidates,strategy,hierarchical,cached[s['id']][0],cached[s['id']][2]) for s in samples}
            dev = [(float(ranks[s['id']][1][ranks[s['id']][0][0]]),s['expected'] is not None)
                   for s in samples if s['split']=='dev']
            threshold = calibrate(dev)
            key = strategy+'/'+('hierarchical' if hierarchical else 'global')
            thresholds[key] = dict(threshold=threshold,dev_n=len(dev),
                                   objective='balanced known acceptance + unknown rejection; tie: higher threshold')
            for s in samples:
                q,qt,f,ft=cached[s['id']]
                for method in ['forced','threshold','numeric','combined']:
                    row=make_record(s,candidates,strategy,hierarchical,ranks[s['id']],threshold,method,qt,ft)
                    rows.append(row)
                    if not args.quiet and s['split']=='test' and strategy==args.strategy and hierarchical==args.hierarchical and method==('combined' if args.guard else 'forced'):
                        display(row)
    summaries=[]
    for strategy in 'ABC':
        for retrieval in ['global','hierarchical']:
            for method in ['forced','threshold','numeric','combined']:
                subset=[r for r in rows if r['split']=='test' and r['strategy']==strategy and r['retrieval']==retrieval and r['method']==method]
                summaries.append(dict(strategy=strategy,retrieval=retrieval,method=method,**metrics(subset)))
    write_csv(Path(args.csv),rows)
    write_csv(OUT/'summary.csv',summaries)
    dump_json(OUT/'thresholds.json',thresholds)
    dump_json(OUT/'precompute.json',precompute)
    zrows=zero_shot(engine,candidates,samples)
    env=environment(engine)
    dump_json(OUT/'environment.json',env)
    dump_json(OUT/'summary.json',summaries)
    category_rows=[]
    confusion={}
    for strategy in 'ABC':
        for retrieval in ['global','hierarchical']:
            for method in ['forced','threshold','numeric','combined']:
                subset=[r for r in rows if r['split']=='test' and r['strategy']==strategy and r['retrieval']==retrieval and r['method']==method]
                for category in sorted(set(r['category'] for r in subset)):
                    category_rows.append(dict(strategy=strategy,retrieval=retrieval,method=method,category=category,
                                              **metrics([r for r in subset if r['category']==category])))
                if method=='forced':
                    for r in subset:
                        if r['expected']:
                            key=(strategy,retrieval,r['expected_enum'],r['field_predicted'])
                            confusion[key]=confusion.get(key,0)+1
    write_csv(OUT/'by_category.csv',category_rows)
    write_csv(OUT/'field_confusion.csv',[dict(strategy=k[0],retrieval=k[1],expected_enum=k[2],predicted_enum=k[3],count=v) for k,v in confusion.items()])
    report(env,summaries,rows,zrows,precompute)
    print('\nHeld-out test summary (known top1 / unknown false acceptance):')
    for s in summaries:
        print(f"{s['strategy']} {s['retrieval']:12s} {s['method']:9s} {s['top1_accuracy']:.2%} / {s['false_acceptance_rate']:.2%}")


def report(env, summaries, rows, zrows, precompute):
    """Build an evidence-linked Chinese report exclusively from actual outputs."""
    pct=lambda x:f'{x:.2%}'
    lines=['# BGE-small-en-v1.5 Proto 语义匹配实测报告',
           '', '实验日期：2026-11-02（Asia/Hong_Kong）。', '',
           '## 范围与实验完整性', '',
           '本实验使用合成相机 Proto；没有真实产品协议或生产查询。所有标准答案在推理前冻结，未增训模型。'
           '184 条已支持测试样本来自 46 个成员 × 4 个固定模板；另有 40 条人工编写的合成未知样本。'
           '开发集为 92 条已支持 + 20 条未知，共 112 条；测试集 224 条。'
           '模板来源逐行记录，不能视为 184 个独立业务意图，也没有声称经过用户或第三方人工审核。', '',
           '开发集和测试集查询不重叠；正例共享候选标签但模板不同，属于域内表达留出。'
           '未知数值与负例表达分离。阈值只按开发集已知接受率与未知拒绝率的平均值标定，'
           '各 A/B/C 与全局/层级配置独立标定，同分取较高阈值；测试集从不参与选择。', '',
           '## 部署与可复现信息', '',
           f"- 模型：BAAI/bge-small-en-v1.5；revision `{env['model_revision']}`。",
           '- 官方直连失败，使用 hf-mirror.com 的固定 revision；权重和数据 SHA256 见 results/environment.json。',
           f"- Python {env['python']}；设备 {env['device']}，{env['gpu']}；PyTorch {env['versions']['torch']}；CUDA build {env['cuda_build']}。",
           f"- Sentence Transformers {env['versions']['sentence-transformers']}；Transformers {env['versions']['transformers']}。",
           f"- {env['parameter_count']:,} 参数；{env['embedding_dimension']} 维；最大长度 {env['max_sequence_length']} tokens。",
           '- 完全本地 safetensors 加载，local_files_only=True；HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1。',
           '- 短对称参数匹配不加 retrieval instruction；A/B/C 共用同一查询向量。L2 归一化 float32 点积即余弦相似度；分数不是概率。', '',
           '## 编码策略与规则', '',
           '- A：原始 Proto 成员名；字段层使用原始 enum 类型名。',
           '- B：去公共 enum 前缀、下划线/CamelCase/数字到单位拆分，保留 Type；不拆复合词。',
           '- C：在 B 上移除字段 Type；slowmotion→Slow Motion，videonormal→Normal Video，timelapse→Time Lapse；daylight 保留，mjpeg 保留大写；MINUS_/PLUS_/ZERO_ 规范成有符号物理数值。',
           '- C 没有加入用户同义表达、答案查表或 acutance→sharpness、PLUS→HIGH 等规则；上述规则是人工先验，C 的收益不能全部归因于 Embedding 泛化。',
           '- 帧率、分辨率、ISO、曝光数值来自名称，绝不使用 wire ID 作物理值；不同 enum 即使 wire ID 相同也视为不同答案。', '',
           '## 独立测试集总表', '',
           'Top1 为已支持样本的最终正确接受率（已知误拒也计错）；Top3 为纯排名召回。字段准确率统计已支持样本。'
           'FAR 为 40 条未知被接受的比例，未知拒绝率=1−FAR。numeric 是额外规则对照，不能计入纯 Embedding。', '',
           '|编码|检索|决策|Top1|Top3|字段|FAR|未知拒绝|已知误拒|P50 ms|P95 ms|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append('|'+ '|'.join([s['strategy'],s['retrieval'],s['method'],pct(s['top1_accuracy']),pct(s['top3_recall']),pct(s['field_accuracy']),pct(s['false_acceptance_rate']),pct(s['unknown_rejection_recall']),pct(s['known_rejection_rate']),f"{s['p50_latency_ms']:.3f}",f"{s['p95_latency_ms']:.3f}"])+'|')
    lines += ['', '仅由 112 条开发集样本标定的余弦阈值：', '',
              '|编码|全局阈值|两级阈值|','|---|---:|---:|']
    for strategy in 'ABC':
        thresholds=[next(r['threshold'] for r in rows if r['strategy']==strategy and r['retrieval']==mode) for mode in ['global','hierarchical']]
        lines.append(f'|{strategy}|{thresholds[0]:.6f}|{thresholds[1]:.6f}|')
    lines += ['', 'forced=纯 Embedding Top1 强制；threshold=开发集阈值；numeric=数值全候选精确校验，非数值无法由该规则判定未知；combined=数值校验+阈值（CLI --guard）。', '',
              '层级第一级仅编码输入字段，选 enum；第二级使用完整 field:value 查询在该 enum 内排名。全局字段预测为 Top1 所属 enum。'
              '字段层误判会封死正确成员；数值 guard 也只验证模型预测的 enum，不能证明字段选择正确。', '',
              '## 144 FPS 关键证据', '',
              '|编码|检索|Top1|余弦|Top2|余弦|Top3|余弦|Top1-2差|',
              '|---|---|---|---:|---|---:|---|---:|---:|']
    for r in rows:
        if r['split']=='test' and r['value']=='144 FPS' and r['method']=='forced':
            lines.append(f"|{r['strategy']}|{r['retrieval']}|{r['top1_name']}|{r['top1_score']:.6f}|{r['top2_name']}|{r['top2_score']:.6f}|{r['top3_name']}|{r['top3_score']:.6f}|{r['margin']:.6f}|")
    lines += ['', '即使分数很高或 margin 较大，也不代表数值相等。数值 guard 在预测 enum 的全部成员中寻找规范化数值完全一致项；144 与 1440 不等。'
              '该校验不支持单位换算、范围、复杂分数；词语数词未转数字。无数字时 numeric 方法不作拒识，combined 仍依靠阈值，因此 UNKNOWN 并无绝对保证。', '',
              '其他数值/符号错误（C，全局，已知样本）：', '',
              '|输入|标准答案|纯 Top1|余弦|','|---|---|---|---:|']
    for r in rows:
        if r['split']=='test' and r['strategy']=='C' and r['retrieval']=='global' and r['method']=='forced' and r['category']=='numeric' and not r['correct']:
            lines.append(f"|{r['field']}: {r['value']}|{r['expected']}|{r['top1_name']}|{r['top1_score']:.6f}|")
    lines += ['', '曝光 +1/+2 被匹配到负值、H.265 被匹配到 H.264，说明不仅不存在的数值会出错，已支持值的精确符号/数字也可能被相似语义淹没。'
              'ISO 的错误同时包含字段误判；只在错误字段内做数值校验会误拒，因此数值 guard 不能替代正确的字段识别。', '',
              '## 按类别分析（C，全局，纯 Embedding）', '',
              '|类别|数量|正确率（已知）或未知错误接受率|', '|---|---:|---:|']
    subset=[r for r in rows if r['strategy']=='C' and r['retrieval']=='global' and r['method']=='forced' and r['split']=='test']
    for category in sorted(set(r['category'] for r in subset)):
        group=[r for r in subset if r['category']==category]
        m=metrics(group)
        val=m['top1_accuracy'] if m['known'] else m['false_acceptance_rate']
        lines.append(f'|{category}|{len(group)}|{pct(val)}|')
    lines += ['', '组合 guard 仍错误接受的未知输入（C，全局）：', '',
              '|输入|错误接受|余弦|','|---|---|---:|']
    for r in rows:
        if r['split']=='test' and r['strategy']=='C' and r['retrieval']=='global' and r['method']=='combined' and not r['expected'] and not r['rejected']:
            lines.append(f"|{r['field']}: {r['value']}|{r['predicted']}|{r['score']:.6f}|")
    lines += ['', '## 零样本候选扩展', '',
              '同一个已加载模型在新增前后的候选集合上重复推理，模型权重未变。新增前排除 1440FPS 与 ImageTextureType，新增后只重算候选向量；'
              '这里的“零样本”仅指本地无增训，不能证明预训练语料从未包含这些词或概念。', '',
              '|场景|阶段|编码|正确数/总数|', '|---|---|---|---:|']
    for scenario in ['new_member','new_enum','naming']:
        for stage in sorted({r['stage'] for r in zrows if r['scenario']==scenario}):
            for strategy in 'ABC':
                group=[r for r in zrows if r['scenario']==scenario and r['stage']==stage and r['strategy']==strategy]
                lines.append(f"|{scenario}|{stage}|{strategy}|{sum(r['correct'] for r in group)}/{len(group)}|")
    lines += ['', '命名变体实验使用两个分开的候选空间，事先在 data/naming_variants.json 中约定 SHARPNESS_HIGH 与 IMAGE_ACUTANCE_PLUS 均表示强边缘增强。'
              '此等价关系仅适用于本合成 fixture，不能推断真实设备也等价；并未根据预测倒推答案。', '',
              '## 运行成本与时延口径', '',
              f"- 模型目录 {env['model_directory_mb']:.2f} MiB；RAM 总量 {env['ram_total_mb']:.0f} MiB。",
              f"- 独立虚拟环境目录 {env['venv_directory_mb']:.1f} MiB（包括 CUDA 库）；下载 wheel/分段临时文件不计入，见 downloads 目录。",
              f"- 模型加载 {env['model_load_ms']:.1f} ms（不含 Python/依赖 import，OS 缓存状态未控制）；3 次预热合计 {env['warmup_ms']:.1f} ms。",
              f"- 加载阶段进程 RSS 增量 {env['model_rss_delta_mb']:.1f} MiB；加载后总 RSS {env['rss_after_load_mb']:.1f} MiB；预热后 {env['rss_after_warmup_mb']:.1f} MiB。",
              f"- CUDA tensor allocated {env['cuda_allocated_mb']:.1f} MiB；reserved {env['cuda_reserved_mb']:.1f} MiB；峰值 allocated {env['cuda_peak_allocated_mb']:.1f} MiB。",
              '- RSS 包含 tokenizer/Python 开销；CUDA allocated 不含驱动和 CUDA context，不能相加理解为独占系统显存。',
              f"- 每条完整查询与字段分别顺序编码 {env['repeats']} 次，先做 3 次预热；CUDA 同步计时。不同策略复用同一查询向量，消除查询编码波动。",
              '- 表中 P50/P95 为各样本重复编码时延中位数的分位数；单条全局时延=完整查询编码+相似度/排序；层级另加字段编码。'
              '没有把字段缓存收益或批量吞吐当作单条时延。raw repeat 时间见 CSV。',
              '- 测试使用正在运行桌面程序的共享笔记本 GPU，未锁定频率/功耗、未控制其他进程；时延是本机观测值，不能直接当成生产 SLA。',
              '- 相似度含字段点积、全候选点积、筛选与排序；层级实现仍计算全部候选点积，因此此 P0 不宣称减少了向量计算量。'
              '数值 guard/阈值的 Python 决策另记 decision_ms，并纳入端到端时延；不含文件输出和进程启动。', '',
              '|编码|候选预计算 ms|字段预计算 ms|向量字节数|', '|---|---:|---:|---:|']
    for p in precompute:
        lines.append(f"|{p['strategy']}|{p['candidate_ms']:.3f}|{p['field_ms']:.3f}|{p['vector_bytes']}|")
    lines += ['', '查询时延分解（C 配置，逐样本重复中位数的分位数）：', '',
              '|检索|组件|P50 ms|P95 ms|','|---|---|---:|---:|']
    import numpy as np
    for retrieval in ['global','hierarchical']:
        group=[r for r in rows if r['split']=='test' and r['strategy']=='C' and r['retrieval']==retrieval and r['method']=='combined']
        for component in ['query_encode_ms','field_encode_ms','similarity_ms','decision_ms']:
            vals=[r[component] for r in group]
            lines.append(f'|{retrieval}|{component}|{np.percentile(vals,50):.4f}|{np.percentile(vals,95):.4f}|')
    lines += ['', '## 典型成功和失败', '', '|输入|标准答案|纯 Top1|分数|结果|', '|---|---|---|---:|---|']
    successes=[r for r in subset if r['expected'] and r['correct']]
    failures=[r for r in subset if r['expected'] and not r['correct']]
    unknowns=[r for r in subset if not r['expected']]
    for r in successes[:4]+failures[:8]+unknowns[:4]:
        lines.append(f"|{r['field']}: {r['value']}|{r['expected'] or 'UNKNOWN'}|{r['top1_name']}|{r['top1_score']:.6f}|{'PASS' if r['correct'] else 'FAIL'}|")
    pure=[s for s in summaries if s['method']=='forced']
    best=max(pure,key=lambda s:s['top1_accuracy'])
    cglobal={s['method']:s for s in summaries if s['strategy']=='C' and s['retrieval']=='global'}
    chier=next(s for s in summaries if s['strategy']=='C' and s['retrieval']=='hierarchical' and s['method']=='forced')
    lines += ['', '## 结论与下一步', '',
              '**未经增训的 BGE-small-en-v1.5 可以提供英文参数的候选召回，但单独不足以安全支撑完整的自然语言到设备枚举自动映射。**', '',
              f"在本合成留出集，最佳纯 Embedding 配置为 {best['strategy']}/{best['retrieval']}，已知 Top1 {pct(best['top1_accuracy'])}；"
              f"其未知错误接受率仍为 {pct(best['false_acceptance_rate'])}，因为强制 Top1 没有表达“不支持”的机制。",
              f"C/global 的阈值方法把未知 FAR 降至 {pct(cglobal['threshold']['false_acceptance_rate'])}，"
              f"但已知正确接受率降至 {pct(cglobal['threshold']['top1_accuracy'])}。"
              f"组合 guard 的 FAR 为 {pct(cglobal['combined']['false_acceptance_rate'])}，"
              f"已知正确接受率 {pct(cglobal['combined']['top1_accuracy'])}，已知误拒 {pct(cglobal['combined']['known_rejection_rate'])}，仍不适合无兜底自动执行。",
              f"层级 C 的字段准确率为 {pct(chier['field_accuracy'])}，低于全局的 {pct(cglobal['forced']['field_accuracy'])}；"
              f"最终 Top1 为 {pct(chier['top1_accuracy'])}，因此本实验未证明两级匹配更好。",
              '精确数值与支持性属于离散约束，余弦距离不能证明相等；词义近似、字段混淆与不规范复合命名又会引入排名错误。'
              'A/B/C 的差别说明候选表述质量有影响，但 C 含人工先验；层级字段预测也会传播错误。', '',
              '建议下一阶段先获取真实设备 Proto、成员含义说明和独立人工标注查询，复验字段混淆及负例覆盖。'
              '将 Embedding 定位为候选召回，保留已支持集合、物理数值/单位验证与 UNKNOWN 返回；非数值未知需补充代表性负例再标定。'
              '本轮没有比较更大模型，不能断言换大模型会解决数值相等或未知检测问题，也不据此直接推荐。', '',
              '## 原始证据', '',
              '- results.csv：全部 dev/test × 3 编码 × 2 检索 × 4 策略，共 8064 行；保留原始 Top3 与最终决策。',
              '- results/summary.csv、summary.json：独立测试集 24 个配置的指标。',
              '- results/thresholds.json：6 个仅由开发集选择的阈值。',
              '- results/zero_shot.csv：插入候选前后与命名变体实测。',
              '- results/environment.json、precompute.json：环境、哈希、成本与预计算。',
              '- data/dataset.json、camera.proto：固定标签与协议样例。', '',
              '## 资料', '',
              '- [模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5) （短对称匹配不必加检索 instruction，余弦不能当概率）。',
              '- [PyTorch CUDA 12.1 2.5.1 分发](https://pytorch.org/get-started/previous-versions/)。',
              '- 本报告全部实验数值来自本地执行，公开模型卡不是评测结果来源。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    """Expose inspect, full evaluation, and one-query offline inference modes."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--csv',default=str(ROOT/'results.csv'))
    parser.add_argument('--hierarchical',action='store_true')
    parser.add_argument('--guard',action='store_true',help='numeric full-enum check + dev threshold')
    parser.add_argument('--strategy',choices=list('ABC'),default='C')
    parser.add_argument('--field')
    parser.add_argument('--value')
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--proto',type=Path,default=PROTO)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('--repeats must be >= 1')
    candidates=parse_proto(args.proto.read_text(encoding='utf-8'))
    if args.eval and args.proto.resolve()!=PROTO.resolve():
        parser.error('Bundled evaluation labels apply only to data/camera.proto; use custom Proto for interactive ranking')
    if args.inspect:
        print(json.dumps(dict(model=str(args.model),revision=(args.model/'REVISION').read_text().strip(),
                              members=len(candidates),enums=len(set(c.enum for c in candidates)),
                              samples=len(json.loads(DATA.read_text())),rules=COMPOUNDS),indent=2))
        for c in candidates:
            print(c.enum,c.name,c.number,' | '.join(describe(c,s) for s in 'ABC'))
        # Inspect verifies real offline loading and inference as well as parsing.
        engine=Engine(args.model,args.device,args.repeats)
        vector=engine.encode('Camera Mode: Slow Motion')
        print(f'Offline load verified; device={engine.device}; embedding={vector.shape}; norm={engine.np.linalg.norm(vector):.6f}')
        return
    if not args.eval and (args.field is None or args.value is None):
        parser.error('Use --inspect, --eval, or both --field and --value')
    engine=Engine(args.model,args.device,args.repeats)
    if args.eval:
        samples=json.loads(DATA.read_text(encoding='utf-8'))
        evaluate(engine,candidates,samples,args)
    else:
        sample=dict(id='interactive',split='interactive',field=args.field,value=args.value,
                    expected=None,expected_enum=None)
        q,qt=engine.timed_encode(args.field+': '+args.value)
        f,ft=engine.timed_encode(args.field)
        ranked=engine.rank(candidates,args.strategy,args.hierarchical,q,f)
        key=args.strategy+'/'+('hierarchical' if args.hierarchical else 'global')
        if args.guard:
            threshold_file=OUT/'thresholds.json'
            if not threshold_file.exists():
                parser.error('Run --eval first to calibrate thresholds on dev data')
            if args.proto.resolve()!=PROTO.resolve():
                parser.error('--guard thresholds are calibrated only for the bundled Proto fixture')
            threshold=json.loads(threshold_file.read_text())[key]['threshold']
        else:
            threshold=0
        display(make_record(sample,candidates,args.strategy,args.hierarchical,ranked,threshold,
                            'combined' if args.guard else 'forced',qt,ft))


if __name__=='__main__':
    main()
