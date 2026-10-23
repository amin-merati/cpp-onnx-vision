# Recorded CPU sample

This is one measured run on 23 October 2026 in a shared Linux x86-64 container,
not a controlled hardware comparison or an optimisation speedup claim.

| Setting | Value |
|---|---|
| CPU reported by environment | Intel Xeon Platinum 8573C |
| Compiler | GCC 13.3.0, CMake Release build |
| ONNX Runtime | 1.20.1, CPU provider |
| Model | ONNX Model Zoo SqueezeNet 1.1, opset 7 |
| Input | 224 x 224 RGB, batch 1 |
| Intra-op / inter-op threads | 1 / 1 |
| Warmup / measured runs | 10 / 100 |
| Median inference call | 6.312 ms |
| p95 inference call | 10.254 ms |
| Image read + preprocessing (single sample) | 6.773 ms |
| Session load (single sample) | 48.137 ms |

The full logits and top-5 output are in `cpu-squeezenet.json`. The top class was
258 (Samoyed), with softmax score 0.7733. The input was the dog example from
https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg, converted
with `scripts/to_ppm.py`. The photograph is not redistributed in this repo.

Command, after downloading/converting that image:

```bash
./build/vision_infer --model models/squeezenet1.1-7.onnx \
  --image data/dog.ppm --runs 100 --warmup 10 --threads 1 \
  --output benchmarks/cpu-squeezenet.json
```

Timing includes the synchronous inference call and host output copy, excluding
preprocessing and top-5 computation. Container contention affects the tail;
repeat on target hardware before using these figures for any sizing decision.
No GPU performance measurements were taken.

Validation also compared the same model's full output against Python ONNX
Runtime on four deterministic image shapes. Maximum absolute logit error was
below 2e-8; preprocessing agreement was within 2e-6. These checks measure
implementation parity, not dataset accuracy.
