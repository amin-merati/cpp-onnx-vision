"""Fetch pinned CPU SDK and optional model; verify SHA-256 before use."""
import argparse
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'sdk': ('https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/'
            'onnxruntime-linux-x64-1.20.1.tgz',
            '67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105'),
    'model': ('https://media.githubusercontent.com/media/onnx/models/main/validated/'
              'vision/classification/squeezenet/model/squeezenet1.1-7.onnx',
              '1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088'),
}


def fetch(kind):
    url, expected = ASSETS[kind]
    request = urllib.request.Request(url, headers={'User-Agent': 'cpp-onnx-vision/0.1'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f'{kind}: checksum mismatch ({actual}); nothing installed')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store_true', help='Also download SqueezeNet 1.1')
    args = parser.parse_args()
    data = fetch('sdk')
    deps = ROOT/'deps'
    deps.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        archive.extractall(deps, filter='data')
    print('Verified ONNX Runtime 1.20.1 Linux x64 CPU SDK')
    if args.model:
        data = fetch('model')
        models = ROOT/'models'
        models.mkdir(exist_ok=True)
        (models/'squeezenet1.1-7.onnx').write_bytes(data)
        print('Verified SqueezeNet 1.1 ONNX model')


if __name__ == '__main__':
    main()
