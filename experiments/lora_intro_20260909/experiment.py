"""本地教学实验：复用 MiniMind 模型、SFT mask 和 LoRA；固定数据与评估协议。"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_CACHE'] = str(HERE / 'cache')
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from safetensors.torch import load_file
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, save_lora, load_lora
from dataset.lm_dataset import SFTDataset

SEED = 42
MAX_LEN = 160
BATCH = 8
LR = 1e-4
GEN = dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
           repetition_penalty=1.0, max_new_tokens=8, use_cache=True, eos_token_id=2)

def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def read(name):
    return json.loads((HERE / name).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)

def base(device):
    c = json.loads((ROOT / 'minimind-3/config.json').read_text())
    c = {k: v for k, v in c.items() if k not in ['model_type', 'architectures']}
    m = MiniMindForCausalLM(MiniMindConfig(**c))
    sd = load_file(ROOT / 'minimind-3/model.safetensors')
    # safetensors 只存一份共享参数；仅恢复已验证的共享别名，其他不匹配必须报错。
    assert m.config.tie_word_embeddings
    assert 'lm_head.weight' not in sd and 'model.embed_tokens.weight' in sd
    sd['lm_head.weight'] = sd['model.embed_tokens.weight']
    m.load_state_dict(sd, strict=True)
    return m.to(device).float()

def attach(m):
    apply_lora(m, rank=16)
    for n, p in m.named_parameters():
        p.requires_grad_('.lora.' in n)
    return [p for p in m.parameters() if p.requires_grad]

class FixedSFT:
    """固定原仓库随机增强的结果；模板和标签扫描直接复用原实现。"""
    def __init__(self, rows, tok):
        self.helper = object.__new__(SFTDataset)
        self.helper.tokenizer = tok
        self.helper.max_length = MAX_LEN
        self.helper.bos_id = tok(f'{tok.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.helper.eos_id = tok(f'{tok.eos_token}\n', add_special_tokens=False).input_ids
        self.examples = []
        for row in rows:
            prompt = self.helper.create_chat_prompt(row['conversations'])
            ids = tok(prompt, add_special_tokens=False).input_ids
            assert len(ids) <= MAX_LEN, '不截断标签；超长样本在划分前过滤'
            labels = self.helper.generate_labels(ids)
            assert any(x != -100 for x in labels[1:])
            assert ids[-2:] == self.helper.eos_id
            self.examples.append((ids, labels))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        return self.examples[i]

def collate(rows):
    length = max(len(x) for x, _ in rows)
    # 右侧 padding + 因果 attention：有效 token 看不到未来 padding。
    # padding 的 label 显式为 -100，与 attention 的因果 mask 是两件事。
    return (torch.tensor([x + [0] * (length-len(x)) for x, _ in rows]),
            torch.tensor([y + [-100] * (length-len(y)) for _, y in rows]))

def batches(rows, tok, shuffle=False):
    return DataLoader(FixedSFT(rows, tok), batch_size=BATCH, shuffle=shuffle,
                      collate_fn=collate, num_workers=0,
                      generator=torch.Generator().manual_seed(SEED))

def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()

def frozen_hash(m):
    h = hashlib.sha256()
    for n, p in m.named_parameters():
        if '.lora.' not in n:
            h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare():
    tok = tokenizer()
    source = HERE / 'data/ChnSentiCorp_htl_all.csv'
    grouped = {}; counts = {'raw': 0, 'empty': 0, 'too_long': 0, 'duplicates': 0, 'conflicting_groups': 0}
    for row in csv.DictReader(source.open(encoding='utf-8-sig')):
        counts['raw'] += 1
        text = unicodedata.normalize('NFKC', row['review']).strip()
        key = re.sub(r'\W+', '', text).lower()
        if not key or row['label'] not in ('0', '1'):
            counts['empty'] += 1; continue
        grouped.setdefault(key, []).append((text, row['label']))
    pools = {'0': [], '1': []}
    for key, group in grouped.items():
        counts['duplicates'] += len(group)-1
        if len({label for _, label in group}) != 1:
            counts['conflicting_groups'] += 1; continue
        text, label = group[0]
        item = {'id': hashlib.sha256(key.encode()).hexdigest(), 'label': label,
                'conversations': [{'role':'user', 'content': '判断下面酒店评论的情感，只输出“正面”或“负面”。\n评论：'+text},
                                  {'role':'assistant', 'content':'正面' if label == '1' else '负面'}]}
        s = tok.apply_chat_template(item['conversations'], tokenize=False, add_generation_prompt=False)
        if len(tok(s, add_special_tokens=False).input_ids) > MAX_LEN:
            counts['too_long'] += 1; continue
        pools[label].append(item)
    rng = random.Random(SEED)
    for rows in pools.values(): rng.shuffle(rows)
    splits = {}
    for name, start, stop in [('train',0,500),('validation',500,550),('test',550,650)]:
        rows = pools['0'][start:stop] + pools['1'][start:stop]
        assert len(rows) == (stop-start)*2
        rng.shuffle(rows); splits[name] = rows
        with (HERE/f'data/{name}.jsonl').open('w') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    ids = [r['id'] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids))
    # 训练前固定的微型通用能力探针；不用于调参，不是标准能力基准。
    probes = [
      ('math','1加1等于几？只输出数字。','2'),
      ('math','3乘以4等于几？只输出数字。','12'),
      ('math','10减去7等于几？只输出数字。','3'),
      ('knowledge','中国的首都是哪里？','北京'),
      ('knowledge','一周有多少天？','七天'),
      ('translation','把英文单词 apple 翻译成中文。','苹果'),
      ('translation','把“你好”翻译成英文。','Hello'),
      ('instruction','请原样重复下面的内容，不要添加解释：春天来了','春天来了'),
      ('instruction','请按从小到大的顺序排列：3、1、2。','1、2、3'),
      ('explanation','请用一句话解释什么是光合作用。','植物利用光能，将二氧化碳和水转化为有机物，并释放氧气。'),
      ('explanation','为什么白天天空通常是蓝色的？','大气分子对阳光中蓝色光的散射比红色光更强。'),
      ('coding','请写一个 Python 函数，返回两个数的和。','def add(a, b):\n    return a + b')]
    general = [{'id':f'general_{i}', 'category':c, 'conversations':[{'role':'user','content':q},{'role':'assistant','content':a}]} for i,(c,q,a) in enumerate(probes)]
    write('data/general.json',general)
    protected = [ROOT/'README.md', ROOT/'requirements.txt', *ROOT.glob('model/*.py'), *ROOT.glob('model/*.json'), *ROOT.glob('trainer/*.py'), *ROOT.glob('dataset/*.py'), *ROOT.glob('scripts/*.py'), ROOT/'eval_llm.py', ROOT/'cyberdyne-blue.json', *ROOT.glob('minimind-3/*.json'), ROOT/'minimind-3/model.safetensors']
    config = dict(seed=SEED,epochs=1,batch_size=BATCH,accumulation_steps=1,max_seq_len=MAX_LEN,
                  learning_rate=LR,optimizer='AdamW',weight_decay=0.0,grad_clip=1.0,
                  rank=16,scaling=1,targets=['q_proj','o_proj'],dtype='float32',
                  generation=GEN,general_max_new_tokens=64,
                  augmentation='no added system; keep empty think in all samples',
                  source='https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/ChnSentiCorp_htl_all',
                  source_sha256=sha(source), counts=counts,pool_counts={k:len(v) for k,v in pools.items()},
                  split_counts={k:len(v) for k,v in splits.items()},
                  base='minimind-3/model.safetensors',base_training_stage='released chat checkpoint; exact SFT/RL lineage unverified',
                  git_commit=None,packages={p:importlib.metadata.version(p) for p in ['torch','transformers','datasets','safetensors','numpy']},
                  original_sha256={str(p.relative_to(ROOT)):sha(p) for p in protected},
                  split_sha256={n:sha(HERE/f'data/{n}.jsonl') for n in splits})
    write('config.json',config)
    # 真正的一条训练样本：逐 token 保存未 shift 标签及监督位置。
    row = splits['train'][0]; ds = FixedSFT([row],tok); x,y = ds[0]
    trace = dict(sample=row,chat=ds.helper.create_chat_prompt(row['conversations']),
                 input_shape=[1,len(x)], hidden_shape=[1,len(x),768],logits_shape=[1,len(x),6400],
                 supervised_tokens=sum(t != -100 for t in y[1:]),
                 rows=[dict(position=i,token_id=t,token=tok.convert_ids_to_tokens(t),label=y[i],
                            predicted_by_logit_position=i-1 if y[i] != -100 else None) for i,t in enumerate(x)])
    write('sample_trace.json',trace)
    print(json.dumps({k:config[k] for k in ['counts','pool_counts','split_counts']},ensure_ascii=False),flush=True)
    print('SAMPLE',json.dumps(row,ensure_ascii=False),flush=True)
    print('CHAT',repr(trace['chat']),'SHAPE',trace['input_shape'],'SUPERVISED',trace['supervised_tokens'],flush=True)

def rows(name):
    return [json.loads(line) for line in (HERE/f'data/{name}.jsonl').read_text().splitlines()]

def smoke(device):
    seed(); tok=tokenizer(); m=base(device).eval(); x,y=next(iter(batches(rows('train')[:8],tok))); x,y=x.to(device),y.to(device)
    with torch.no_grad(): initial=m(x,labels=y); ref_logits=initial.logits.detach().clone()
    params=attach(m)
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.0)
    out=m(x,labels=y)
    zero_delta=float((out.logits-ref_logits).abs().max())
    manual=F.cross_entropy(out.logits[:,:-1].reshape(-1,6400),y[:,1:].reshape(-1),ignore_index=-100)
    assert torch.allclose(out.loss,manual)
    assert zero_delta == 0
    before=frozen_hash(m)
    out.loss.backward()
    grad={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad}
    assert all(v == 0 for n,v in grad.items() if '.A.' in n)
    assert any(v > 0 for n,v in grad.items() if '.B.' in n)
    assert all(p.grad is None for n,p in m.named_parameters() if '.lora.' not in n)
    first_loss=float(out.loss.detach()); opt.step(); opt.zero_grad(set_to_none=True)
    del initial,ref_logits,out,manual
    times=[]; losses=[]
    m.train()
    for step in range(6):
        sync(device); start=time.perf_counter()
        out=m(x,labels=y); assert torch.isfinite(out.loss)
        out.loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0)
        opt.step();opt.zero_grad(set_to_none=True);sync(device)
        times.append(time.perf_counter()-start); losses.append(float(out.loss.detach()))
        print(f'SMOKE {step+1}/6 loss={losses[-1]:.5f} seconds={times[-1]:.3f}',flush=True)
    assert before == frozen_hash(m)
    m.eval(); save_lora(m,HERE/'smoke_lora.pth')
    with torch.no_grad(): logits=m(x).logits
    fresh=base(device); attach(fresh);load_lora(fresh,HERE/'smoke_lora.pth');fresh.eval()
    with torch.no_grad(): diff=float((fresh(x).logits-logits).abs().max())
    result=dict(device=device,base_params=63912192,lora_params=sum(p.numel() for p in params),
                targets=[n for n,v in m.named_modules() if hasattr(v,'lora')],
                injection_max_logit_diff=zero_delta,first_loss=first_loss,first_grad_norms=grad,
                frozen_unchanged=True,batch_shape=list(x.shape),warm_seconds=times,
                median_warm_batch_seconds=statistics.median(times[2:]),
                estimated_125_train_steps_seconds=statistics.median(times[2:])*125,
                reload_fp16_max_logit_diff=diff,smoke_losses=losses,
                note='Smoke optimizer/model discarded; formal training starts from base again.')
    assert diff < 0.02
    write('smoke.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

@torch.no_grad()
def nll(m,data,tok,device):
    m.eval(); total=0.; count=0
    for x,y in batches(data,tok):
        x,y=x.to(device),y.to(device); out=m(x,labels=y)
        n=int((y[:,1:] != -100).sum()); total+=float(out.loss)*n;count+=n
    return dict(loss=total/count,supervised_tokens=count)

@torch.no_grad()
def generate(m,data,tok,device,max_new):
    m.eval(); result=[]
    for i,row in enumerate(data):
        prompt=tok.apply_chat_template(row['conversations'][:-1],tokenize=False,add_generation_prompt=True,open_thinking=False)
        x=tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.to(device)
        generated=m.generate(x,**{**GEN,'max_new_tokens':max_new})[0,x.shape[1]:].tolist()
        response=tok.decode(generated,skip_special_tokens=True).strip()
        gold=row['conversations'][-1]['content']
        result.append(dict(id=row['id'],prompt=row['conversations'][0]['content'],reference=gold,
                           output=response,generated_ids=generated,
                           exact=response==gold,hit_limit=len(generated)==max_new and generated[-1]!=2))
        if (i+1)%50 == 0: print(f'GENERATE {i+1}/{len(data)}',flush=True)
    return result

def scores(preds):
    labels=['负面','正面']; f1=[]
    for label in labels:
        tp=sum(p['reference']==label and p['output']==label for p in preds)
        fp=sum(p['reference']!=label and p['output']==label for p in preds)
        fn=sum(p['reference']==label and p['output']!=label for p in preds)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(accuracy=sum(p['exact'] for p in preds)/len(preds),macro_f1=sum(f1)/2,
                valid_label_rate=sum(p['output'] in labels for p in preds)/len(preds),
                output_counts={label:sum(p['output']==label for p in preds) for label in labels},
                hit_limit=sum(p['hit_limit'] for p in preds),n=len(preds))

def evaluate(m,tok,device,name):
    result={split:nll(m,rows(split),tok,device) for split in ['validation','test']}
    general=read('data/general.json');result['general_reference_nll']=nll(m,general,tok,device)
    preds=generate(m,rows('test'),tok,device,8);write(f'{name}_test_outputs.json',preds)
    gen=generate(m,general,tok,device,64);write(f'{name}_general_outputs.json',gen)
    result['test_generation']=scores(preds)
    result['general_exact']=sum(p['exact'] for p in gen)/len(gen)
    write(f'{name}_metrics.json',result);print(name,json.dumps(result),flush=True)
    return result

def train(device):
    assert (HERE/'smoke.json').exists(), '先执行 smoke'
    assert not (HERE/'adapter.pth').exists(), '防止覆盖；新实验请使用新目录'
    seed();tok=tokenizer();m=base(device)
    baseline=evaluate(m,tok,device,'before')
    params=attach(m);frozen=frozen_hash(m)
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.)
    loader=batches(rows('train'),tok,True);total_steps=len(loader);times=[];seen=0
    with (HERE/'train_log.jsonl').open('w') as log:
        for step,(x,y) in enumerate(loader,1):
            m.train();x,y=x.to(device),y.to(device)
            # 与原 get_lr 同样的余弦衰减下限 0.1，仅一个 epoch。
            lr=LR*(0.1+0.45*(1+math.cos(math.pi*step/total_steps)))
            for group in opt.param_groups:group['lr']=lr
            sync(device);start=time.perf_counter();opt.zero_grad(set_to_none=True)
            out=m(x,labels=y);loss=out.loss+out.aux_loss
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.0);assert torch.isfinite(norm)
            opt.step();sync(device);duration=time.perf_counter()-start;times.append(duration);seen+=len(x)
            event=dict(step=step,epoch=1,examples_seen=seen,loss=float(loss.detach()),
                       supervised_tokens=int((y[:,1:]!=-100).sum()),lr=lr,grad_norm=float(norm),
                       batch_shape=list(x.shape),seconds=duration)
            if step%25==0 or step==total_steps:event['validation']=nll(m,rows('validation'),tok,device)
            log.write(json.dumps(event)+'\n');log.flush()
            if step%10==0 or 'validation' in event:print('TRAIN',json.dumps(event),flush=True)
    assert seen==1000 and total_steps==125
    assert frozen==frozen_hash(m)
    m.eval();save_lora(m,HERE/'adapter.pth')
    torch.save({'optimizer':opt.state_dict(),'epoch':1,'step':total_steps,'torch_rng':torch.get_rng_state()},HERE/'optimizer_final.pt')
    # 比较真实磁盘产物：仓库 save_lora 将适配器写成 FP16，内存训练是 FP32。
    probe=rows('test')[:8];x,y=next(iter(batches(probe,tok)));x=x.to(device)
    with torch.no_grad():live=m(x).logits.cpu()
    live_outputs=generate(m,probe,tok,device,8)
    del opt,params,m
    if device=='mps':torch.mps.empty_cache()
    fresh=base(device);attach(fresh);load_lora(fresh,HERE/'adapter.pth');fresh.eval()
    with torch.no_grad():diff=float((fresh(x).logits.cpu()-live).abs().max())
    reload_outputs=generate(fresh,probe,tok,device,8)
    assert diff<0.02
    assert [p['generated_ids'] for p in live_outputs]==[p['generated_ids'] for p in reload_outputs]
    after=evaluate(fresh,tok,device,'after')
    preserved=all(sha(ROOT/p)==v for p,v in read('config.json')['original_sha256'].items());assert preserved
    result=dict(device=device,steps=total_steps,examples=seen,epochs=1,
                train_compute_seconds=sum(times),median_batch_seconds=statistics.median(times[5:]),
                frozen_parameters_unchanged=True,original_files_unchanged=preserved,
                reload_max_logit_diff=diff,reload_greedy_equal_on_8=True,
                adapter_sha256=sha(HERE/'adapter.pth'),before=baseline,after=after)
    write('result.json',result);print('DONE',json.dumps(result),flush=True)

def reload_check(device):
    seed();tok=tokenizer();m=base(device);attach(m);load_lora(m,HERE/'adapter.pth');m.eval()
    actual=generate(m,rows('test')[:8],tok,device,8)
    expected=read('after_test_outputs.json')[:8]
    assert [p['generated_ids'] for p in actual]==[p['generated_ids'] for p in expected]
    write('reload_check.json',dict(fresh_process=True,device=device,matched_examples=8,adapter_sha256=sha(HERE/'adapter.pth')))
    print('Fresh-process reload: 8/8 token sequences match.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','train','reload']);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        if a.device=='mps':assert torch.backends.mps.is_available(), 'MPS requires host access outside sandbox'
        {'smoke':smoke,'train':train,'reload':reload_check}[a.mode](a.device)
