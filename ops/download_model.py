#!/usr/bin/env python3
"""Download a commit-pinned checkpoint using the already-built public image."""
import argparse
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--key', required=True)
parser.add_argument('--image', default='ghcr.io/krisitown/local-jev-api:spark-20261013')
args = parser.parse_args()
pins = json.loads((ROOT / 'references/runtime-recipes/model-pins.json').read_text())
pin = pins[args.key]
download = pin.get('recommended_download_alias', pin)
if download is not pin and download['resolved_sha'] != pin['revision']:
    raise ValueError(f"Download alias for {args.key} does not resolve to its pinned revision")
destination = ROOT / 'checkpoints' / args.key
destination.mkdir(parents=True, exist_ok=True)
code = '''import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir='/checkpoint',
                  ignore_patterns=['assets/*','videos/*','multilingual/*','typed-decisions/*','eval/*'])
'''
command = ['docker', 'run', '--rm', '--name', f'local-jev-bench-download-{args.key}-{time.time_ns()}',
           '--pull=never', '--entrypoint', 'python3',
           '-e', 'HF_HUB_OFFLINE=0', '-e', 'TRANSFORMERS_OFFLINE=0',
           '-e', 'HF_HUB_DISABLE_TELEMETRY=1',
           '-v', f'{destination}:/checkpoint', args.image,
           '-c', code, download['repo'], pin['revision']]
print(json.dumps({'key': args.key, 'repo': pin['repo'], 'download_repo': download['repo'],
                  'revision': pin['revision'], 'destination': str(destination)}), flush=True)
subprocess.run(command, check=True)
