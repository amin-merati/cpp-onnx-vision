"""End-to-end C++/NumPy/ONNX Runtime parity and negative integration tests."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def reference_preprocess(rgb, size):
    """Independent vectorised reference; double-precision interpolation, float32 output."""
    h, w, _ = rgb.shape
    short = int(np.floor(size * 256 / 224 + 0.5))
    rw, rh = np.floor(np.array([w, h]) * short / min(w, h) + 0.5).astype(int)
    xs = np.clip((np.arange(size) + (rw-size)//2 + .5) * w/rw - .5, 0, w-1)
    ys = np.clip((np.arange(size) + (rh-size)//2 + .5) * h/rh - .5, 0, h-1)
    x0, y0 = xs.astype(int), ys.astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    wx, wy = (xs-x0)[None, :, None], (ys-y0)[:, None, None]
    a, b = rgb[y0[:, None], x0], rgb[y0[:, None], x1]
    c, d = rgb[y1[:, None], x0], rgb[y1[:, None], x1]
    pixels = (1-wy)*((1-wx)*a+wx*b) + wy*((1-wx)*c+wx*d)
    normalised = (pixels/255 - np.array([.485,.456,.406])) / np.array([.229,.224,.225])
    return normalised.transpose(2,0,1)[None].astype(np.float32)


def fixture(path, shape=(1, 3, 8, 8)):
    # Deliberately tiny deterministic test graph, not a trained vision model.
    weights = np.array([[.7,-.2,.1,.4],[-.3,.5,.9,.2],[.6,.8,-.4,.3]], dtype=np.float32)
    bias = np.array([.2,-.1,.3,0], dtype=np.float32)
    nodes = [helper.make_node('ReduceMean', ['image'], ['means'], axes=[2,3], keepdims=0),
             helper.make_node('MatMul', ['means','weight'], ['scores']),
             helper.make_node('Add', ['scores','bias'], ['logits'])]
    graph = helper.make_graph(nodes, 'deterministic-parity-fixture',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1,4])],
        [numpy_helper.from_array(weights, 'weight'), numpy_helper.from_array(bias, 'bias')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--model', type=Path, help='Also validate an external square-input classifier')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    model = args.work/'fixture.onnx'
    fixture(model)
    models = [model] + ([args.model] if args.model else [])
    rng = np.random.default_rng(17)
    for index, (h,w) in enumerate([(13,19),(21,9),(1,1),(8,8)]):
        rgb = rng.integers(0, 256, (h,w,3), dtype=np.uint8)
        image = args.work/f'image-{index}.ppm'
        image.write_bytes(f'P6\n{w} {h}\n255\n'.encode() + rgb.tobytes())
        for current_model in models:
            result, tensor = args.work/'result.json', args.work/'input.f32'
            command = [str(args.exe.resolve()), '--model', str(current_model), '--image', str(image),
                       '--runs','3','--warmup','1','--output',str(result),'--dump-input',str(tensor)]
            subprocess.run(command, check=True)
            session = ort.InferenceSession(str(current_model), providers=['CPUExecutionProvider'])
            size = session.get_inputs()[0].shape[2]
            expected_input = reference_preprocess(rgb, size)
            actual_input = np.fromfile(tensor, dtype=np.float32).reshape(expected_input.shape)
            np.testing.assert_allclose(actual_input, expected_input, rtol=0, atol=2e-6)
            expected = session.run(None, {session.get_inputs()[0].name: expected_input})[0].ravel()
            report = json.loads(result.read_text())
            np.testing.assert_allclose(report['logits'], expected, rtol=2e-4, atol=2e-4)
            assert report['top5'][0]['class_id'] == int(expected.argmax())
            assert report['inference_p95_ms'] >= report['inference_median_ms'] > 0
            print(f'PASS {current_model.name}: {w}x{h}, max logit error '
                  f'{np.max(np.abs(np.array(report["logits"])-expected)):.3g}')
    image = args.work/'image-0.ppm'
    base = [str(args.exe.resolve()), '--model', str(model), '--image', str(image), '--runs', '1']
    for extra in [['--runs','0'],['--runs','2x'],['--threads','0'],['--provider','invalid'],
                  ['--image',str(args.work/'missing.ppm')],['--unknown','1']]:
        failure = subprocess.run(base+extra, capture_output=True, text=True)
        assert failure.returncode != 0 and 'error:' in failure.stderr, extra
    invalid = args.work/'invalid-shape.onnx'
    fixture(invalid, (1,3,8,9))
    failure = subprocess.run(base+['--model',str(invalid)], capture_output=True, text=True)
    assert failure.returncode != 0 and 'Expected static' in failure.stderr
    print('PASS invalid CLI values, missing image and unsupported model shape')


if __name__ == '__main__':
    main()
