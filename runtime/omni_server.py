"""Release unreachable CPU loading objects before benchmarking CPU-first Omni.

The checkpoint, native inference, tokenization and scoring are unchanged.
"""
import ctypes
import gc
import time
import server

original_load = server.load

def load_and_release_cpu_staging():
    original_load()
    started = time.perf_counter()
    before = server.mem()
    collected = gc.collect()
    libc = ctypes.CDLL('libc.so.6')
    trim_result = libc.malloc_trim(0)
    server.torch.cuda.synchronize()
    after = server.mem()
    event = {'event': 'cpu_loading_cleanup', 'gc_objects_collected': collected,
             'malloc_trim_result': trim_result, 'before': before, 'after': after,
             'seconds': time.perf_counter()-started}
    server.event(event)
    server.startup['cpu_loading_cleanup'] = event
    server.startup['load_seconds'] += event['seconds']
    server.startup['memory'] = after
    if after['host_available_bytes'] < 16 * 1024**3:
        raise RuntimeError('CPU-first loading still leaves under 16 GiB available; stop before long-input tests')

server.app.router.on_startup[:] = [load_and_release_cpu_staging if fn is original_load else fn
                                  for fn in server.app.router.on_startup]
app = server.app
