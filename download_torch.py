"""Resumable range downloader for the exact official CUDA 12.1 wheel bytes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent / 'downloads'
NAME = 'torch-2.5.1+cu121-cp312-cp312-win_amd64.whl'
URL = 'https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/pytorch-wheels/cu121/torch-2.5.1%2Bcu121-cp312-cp312-win_amd64.whl'
SIZE = 2449331371
CHUNK = 16 * 1024 * 1024


def fetch(index):
    """Download a verified-size range, reusing previously complete segments."""
    start = index * CHUNK
    end = min(start + CHUNK, SIZE) - 1
    path = BASE / f'torch_segment_{index:03}.part'
    if path.exists() and path.stat().st_size == end-start+1:
        return path
    subprocess.run(['curl.exe','--silent','--show-error','-L','--fail','--retry','3',
                    '--max-time','180','--range',f'{start}-{end}',URL,'-o',str(path)],check=True)
    if path.stat().st_size != end-start+1:
        raise ValueError(f'Wrong range size: {path}')
    print(f'Range {index+1}/{(SIZE+CHUNK-1)//CHUNK}',flush=True)
    return path


if __name__ == '__main__':
    BASE.mkdir(exist_ok=True)
    existing=BASE/NAME
    if existing.exists() and existing.stat().st_size==SIZE:
        with existing.open('rb') as src:
            actual=hashlib.file_digest(src,'sha256').hexdigest()
        if actual=='473d76257636c66b22cbfac6f616d6b522ef3d3473c13decb1afda22a7b059eb':
            print('Existing wheel verified against official SHA256: '+actual)
            sys.exit(0)
    with ThreadPoolExecutor(max_workers=12) as pool:
        paths=list(pool.map(fetch,range((SIZE+CHUNK-1)//CHUNK)))
    target=BASE/NAME
    with target.open('wb') as out:
        for path in paths:
            with path.open('rb') as src:
                shutil.copyfileobj(src,out,1024*1024)
    md5=hashlib.file_digest(target.open('rb'),'md5').hexdigest()
    # Same ETag independently returned by official download.pytorch.org and mirror.
    if md5 != 'b141c37f56875eef112278665f54db70':
        raise ValueError('Wheel checksum mismatch: '+md5)
    with target.open('rb') as src:
        sha256=hashlib.file_digest(src,'sha256').hexdigest()
    # Published in https://download.pytorch.org/whl/cu121/torch/ for this wheel.
    if sha256 != '473d76257636c66b22cbfac6f616d6b522ef3d3473c13decb1afda22a7b059eb':
        raise ValueError('Official SHA256 mismatch: '+sha256)
    print(f'Verified {target}: {SIZE} bytes, MD5 {md5}',flush=True)
    print('Official SHA256: '+sha256,flush=True)
