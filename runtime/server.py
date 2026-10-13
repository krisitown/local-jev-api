"""Instrumented, strict no-truncation benchmark API; publisher inference is unchanged."""
import gc
import hashlib
import importlib
import inspect
import json
import os
import pathlib
import sys
import time
from collections import Counter, OrderedDict
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

MODEL_ID=os.environ['MODEL_ID']; MODEL_PATH=os.environ['MODEL_PATH']
BACKEND=os.environ['BACKEND']; REVISION=os.environ['MODEL_REVISION']
app=FastAPI(title='Strict local decision benchmark API')
lock=Lock(); cache=OrderedDict(); ready=False; startup={}; limits={}
predict=None; measure=None

def mem():
    values={x.split(':')[0]:int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines()}
    return {'torch_allocated_bytes':torch.cuda.memory_allocated(),
            'torch_reserved_bytes':torch.cuda.memory_reserved(),
            'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'host_available_bytes':values['MemAvailable'],
            'host_swap_used_bytes':values['SwapTotal']-values['SwapFree']}

def event(data): print(json.dumps(data),flush=True)

@app.on_event('startup')
def load():
    global ready,predict,measure,startup,limits
    start=time.perf_counter(); torch.manual_seed(20261013); torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.85)
    sys.path.insert(0,MODEL_PATH)
    if BACKEND=='clef':
        module=importlib.import_module('joint_schema_model')
        device=os.environ.get('CLEF_LOAD_DEVICE','cuda')
        model,processor=module.load_release_model(MODEL_PATH,device=device)
        if device!='cuda': model.to(device='cuda')
        model.eval()
        cap=int(inspect.signature(module.systemone).parameters['max_length'].default)
        limits={'total_tokens':cap,'native_overflow':'state_prefix_truncation'}
        def measure_request(req):
            enc=module.encode_record(processor.tokenizer,req,max_length=2000000,processor=processor)
            return {'full_tokens':len(enc.input_ids),'within_limits':len(enc.input_ids)<=cap}
        def predict_request(req):
            with torch.inference_mode(): return module.systemone(model,processor,req)
        measure,predict=measure_request,predict_request
    else:
        from backends import load_backend
        predict=load_backend(BACKEND,MODEL_PATH)
        closed=dict(zip(predict.__code__.co_freevars,[c.cell_contents for c in predict.__closure__]))
        if BACKEND in ('jeff','pplx'):
            model=closed['model']; limits={'total_tokens_per_question':8192,'native_overflow':'reject'}
            def measure_request(req):
                counts=[]
                for q in req['questions'].values():
                    batch=model.prepare([{'state':req['state'],'question':q}],max_length=2000000)
                    counts.append(int(batch.input_tokens)); del batch
                return {'full_tokens':sum(counts),'branch_tokens':counts,'within_limits':max(counts)<=8192}
            measure=measure_request
        elif BACKEND=='neohorse':
            engine=closed['engine']; limits={'state_tokens':engine.max_state,'branch_tokens':engine.max_branch,
                                           'expanded_tokens':engine.max_tokens,'native_overflow':'reject'}
            from neohorse_decision._vendor.schema import SystemOneRequest,to_record
            def measure_request(req):
                rec,_=to_record(SystemOneRequest(**{**req,'model':'NeoHorse-Jev-4B'}))
                enc=engine.model.encode(engine.tokenizer,rec,max_state=2000000,max_branch=2000000,strict=True)
                counts=Counter(enc['seg']); state=counts[0]
                branches=[state+v for k,v in counts.items() if k!=0]
                expanded=len(enc['ids'])+(len(req['questions'])-1)*state
                return {'full_tokens':expanded,'state_tokens':state,'branch_tokens':branches,
                        'within_limits':state<=engine.max_state and max(branches)<=engine.max_branch and expanded<=engine.max_tokens}
            measure=measure_request
        elif BACKEND=='laya':
            agent=closed['agent']
            assert agent.device.type=='cuda','CPU fallback is not a comparable GPU run'
            cap=agent.cfg.get('max_len',512); head=agent.cfg.get('head_max_len',192)
            limits={'total_tokens_per_question':cap,'head_tokens':head,'native_overflow':'state_prefix_truncation'}
            from laya.common import build_sequence
            def measure_request(req):
                counts=[]
                for key,value in req['questions'].items():
                    q=agent._to_internal({**value,'instructions':value.get('instructions',key)})
                    seq,_=build_sequence(agent.tok,req['state'],q,max_len=2000000,head_max_len=head)
                    counts.append(len(seq))
                return {'full_tokens':sum(counts),'branch_tokens':counts,'within_limits':max(counts)<=cap}
            measure=measure_request
    torch.cuda.synchronize(); ready=True
    startup={'load_seconds':time.perf_counter()-start,'memory':mem(),'limits':limits,
             'model':MODEL_ID,'revision':REVISION,'backend':BACKEND,
             'torch':torch.__version__,'transformers':importlib.import_module('transformers').__version__,
             'cuda_allocator_fraction_limit':0.85}
    event({'event':'ready',**startup})

def validate(req):
    if not ready: raise HTTPException(503,'not ready')
    if req.get('model',MODEL_ID)!=MODEL_ID: raise HTTPException(400,'model mismatch')
    if set(req)-{'model','state','questions'}: raise HTTPException(400,'unsupported input fields')
    if not isinstance(req.get('state'),str) or not isinstance(req.get('questions'),dict) or not req['questions']:
        raise HTTPException(400,'text state and questions required')

def inspect_unlocked(req):
    digest=hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
    if digest in cache: return cache[digest]
    result={**measure(req),'limits':limits,'state_chars':len(req['state']),
            'state_bytes':len(req['state'].encode()),'request_sha256':digest}
    result['would_truncate']=not result['within_limits'] and limits['native_overflow']=='state_prefix_truncation'
    cache[digest]=result
    if len(cache)>4096: cache.popitem(last=False)
    return result

def inspect_request(req):
    with lock: return inspect_unlocked(req)

@app.get('/health')
def health(): return {'ready':ready,**startup}

@app.get('/metrics')
def metrics(): return mem()

@app.post('/inspect')
async def inspect_endpoint(req:dict):
    validate(req)
    return await run_in_threadpool(inspect_request,req)

def evaluate(req,received):
    with lock:
        acquired=time.perf_counter(); inspection=inspect_unlocked(req)
        if not inspection['within_limits']:
            raise HTTPException(413,{'error':'context_limit_no_truncation','inspection':inspection})
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        before=mem(); start=time.perf_counter()
        try:
            response=predict(req); torch.cuda.synchronize()
            timing={'compute_seconds':time.perf_counter()-start,'queue_seconds':acquired-received}
            response['model']=MODEL_ID
            response['_benchmark']={'inspection':inspection,'memory_before':before,'memory_after':mem(),**timing}
            event({'event':'request','request_sha256':inspection['request_sha256'],**timing,'memory':mem()})
            return response
        except torch.cuda.OutOfMemoryError as exc:
            peak=mem(); gc.collect(); torch.cuda.empty_cache()
            event({'event':'oom','memory':peak,'request_sha256':inspection['request_sha256']})
            raise HTTPException(507,{'error':'cuda_oom','memory':peak}) from exc

@app.post('/v1/systemone')
async def endpoint(req:dict):
    received=time.perf_counter(); validate(req)
    return await run_in_threadpool(evaluate,req,received)
