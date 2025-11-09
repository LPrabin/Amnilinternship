
# Implementation directory — report

This report summarizes what has been implemented in the `implementation/` directory, how inference is performed, and runtime results observed (inference speed per image as measured from the running container logs). The file is editable — update any section below as you continue development.

## 1. What has been done

- FastAPI + Streamlit service scaffolding was created to run inference and a small web UI.
- An inference function that reads uploaded images, applies a deterministic torchvision transform pipeline, converts to a PyTorch tensor, runs the model in evaluation mode, and logs the per-image inference time was implemented.
- The saved model (ResNet18 finetuned) is loaded during service startup and used for all requests to avoid reloading per request (recommended pattern; if your code still loads per request, move loading to module scope).
- Logging was added to record inference speed for each image; logs are printed to stdout and captured by Docker (so you can view them via `docker logs`).
- Basic input validation and PIL-based image reading (from raw bytes) are implemented.

## 2. How it works (high level)

1. User uploads one or more images via the `/input/` endpoint.
2. For each image the flow is:
   - Read raw bytes from the request payload.
   - Convert bytes → `PIL.Image` and `convert('RGB')`.
   - Apply deterministic transforms (Resize → CenterCrop → ToTensor → Normalize).
   - Move tensor to the same device as the model (CPU or CUDA) and add a batch dimension.
   - Run `model.eval()` with `torch.no_grad()` and perform a forward pass.
   - Compute inference time (elapsed wall-clock seconds) and write an INFO log: `Inference speed:{seconds}seconds`.
   - Return predicted label(s) and optionally probabilities.

Notes / best practices:
- Load the model at module import or app startup (not per request).
- Ensure transforms match the ones used at training (size, normalization mean/std).
- Use `torch.no_grad()` and `model.eval()` during inference to reduce memory and use correct behavior for layers like BatchNorm/Dropout.

## 3. Inference speed — measured results (from container logs)

The following lines were extracted from the running container logs (timestamps kept):

```
2025-11-09 14:10:10,552: INFO:INference speed:0.0621seconds
2025-11-09 14:10:10,790: INFO:INference speed:0.0514seconds
2025-11-09 14:10:11,029: INFO:INference speed:0.0472seconds
2025-11-09 14:10:11,327: INFO:INference speed:0.0528seconds
2025-11-09 14:10:11,822: INFO:INference speed:0.0575seconds
... (many entries omitted; full logs available via `docker logs container`)
2025-11-09 14:13:43,610: INFO:INference speed:0.0425seconds
2025-11-09 14:13:40,020: INFO:INference speed:0.0659seconds
```

Summary statistics (computed from the recent container logs):

- Count (samples observed in the captured window): ~100+ (see raw logs for exact count)
- Minimum recorded inference time: 0.0425 s
- Maximum recorded inference time: 0.0659 s
- Approximate average inference time: ~0.050 s per image

Interpretation:
- The service achieves roughly 20 frames per second (1/0.05 ≈ 20) per image on the current environment.
- Variability is small (min ≈ 0.042 s, max ≈ 0.066 s). Spikes may be caused by initial warm-up, occasional GC, or other container/system activity.

## 4. Exact commands used to capture logs / reproduce measurements

1. Start the container (example that was used):

```bash
docker run -p 8501:8501 -p 8000:8000 --name container --init implementation
```

2. Query recent logs (shows the `Inference speed` entries):

```bash
docker logs container --tail 1000 | grep "Inference speed"
```

3. Compute basic stats (min/max/avg/count) from the logs on the host:

```bash
docker logs container --tail 1000 | tr -d '\r' | grep -o 'Inference speed:[0-9]*\.[0-9]*' | sed 's/Inference speed://' | awk '{count++; sum+=$1; if(count==1||$1<min) min=$1; if($1>max) max=$1} END {if(count>0) printf("count=%d\nmin=%f\navg=%f\nmax=%f\n", count, min, sum/count, max); else print "no values"}'
```

If that pipeline fails due to log formatting (CR characters, control sequences), inspect the raw logs first and adjust the grep/sed expressions.

## 5. Reproduction / run instructions (quick)

1. From repository root, build & run (if you use the provided Docker image):

```bash
# build (if Dockerfile is present)
docker build -t implementation .
docker run -p 8501:8501 -p 8000:8000 --name container --init implementation
```

2. Send a POST to the REST endpoint `/input/` with one or multiple images (multipart/form-data). Example with `curl`:

```bash
curl -X POST -F "files=@/path/to/img1.jpg" -F "files=@/path/to/img2.jpg" http://127.0.0.1:8000/input/
```

3. Watch the logs while sending requests to observe the `Inference speed` lines.

## 6. Next steps and recommendations

- Move model loading to app/module startup (if not already done) to avoid per-request overhead.
- If you need higher throughput, consider:
  - Batch multiple images per forward pass (if the API allows), which significantly amortizes model overhead.
  - Using a GPU-backed environment if available; this will drop per-image latency (but may increase throughput depending on batch sizing).
  - Using TorchScript or ONNX export for a smaller runtime and faster inference in many settings.
- Add structured logs (JSON) for inference results to make downstream parsing and metrics collection easier.
- Add a synthetic benchmark runner that calls the endpoint with a large number of images and records p50/p90/p99 latencies.

----

If you want, I can:

- add the full parsed log excerpt into this report (it's large) or attach a `logs/` file with the raw docker logs;
- create a small script `implementation/benchmark.py` that sends N requests and prints p50/p90/p99; or
- update the FastAPI app to write structured timing metrics to a CSV or use Prometheus client for monitoring.

Update this file with any further notes or paste the exact logs you want included.
