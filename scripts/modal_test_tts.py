"""
Modal test script for qwen3_tts CUDA inference on A100.

Usage:
    modal run scripts/modal_test_tts.py

This script:
1. Builds the qwen3_tts binary with --features cuda on an A100 GPU
2. Downloads Qwen3-TTS model weights from HuggingFace
3. Runs inference with RUST_BACKTRACE=1 for diagnostics
4. Persists .luminal_cache/ between runs to skip egglog compilation
"""

import modal

# Persistent volume for model weights (survives across runs)
model_volume = modal.Volume.from_name("qwen3-tts-weights", create_if_missing=True)

# Persistent volume for e-graph compilation cache
cache_volume = modal.Volume.from_name("luminal-egraph-cache", create_if_missing=True)

LOCAL_REPO = "/Users/olety/Desktop/code/oneiron-luminal"
MOUNT_PATH = "/root/luminal"

# Cache directory on the volume
CACHE_DIR = "/cache/luminal_cache"

# Two HF repos needed by the binary (configured via env vars)
MAIN_MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
MAIN_MODEL_DIR = "/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TOKENIZER_DIR = "/models/Qwen3-TTS-Tokenizer-12Hz"

# Start from Modal's Debian base (has proper Python) and add CUDA toolkit
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "curl", "build-essential", "pkg-config", "git", "wget", "gnupg",
        "software-properties-common", "protobuf-compiler",
    )
    .run_commands(
        # Add NVIDIA CUDA repo and install toolkit
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-4",
        # Install Rust toolchain
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .pip_install("huggingface_hub[hf_xet]")
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_PATH": "/usr/local/cuda-12.4",
        "CUDA_HOME": "/usr/local/cuda-12.4",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64",
    })
    .add_local_dir(
        LOCAL_REPO,
        remote_path=MOUNT_PATH,
        ignore=["target/", ".git/", "node_modules/"],
    )
)

app = modal.App("qwen3-tts-cuda-test", image=image)


def download_model(repo_id: str, local_dir: str, patterns: list[str]):
    """Download a HuggingFace model using the Python API."""
    import os
    if os.path.exists(f"{local_dir}/model.safetensors"):
        print(f"  Already cached: {local_dir}", flush=True)
        return True
    print(f"  Downloading {repo_id} -> {local_dir}", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=patterns,
    )
    return os.path.exists(f"{local_dir}/model.safetensors")


def run_streaming(cmd, env=None, cwd=None):
    """Run a command with real-time stdout/stderr streaming."""
    import subprocess
    import sys

    print(f"\n{'='*60}", flush=True)
    print(f"Running: {cmd}", flush=True)
    print(f"{'='*60}", flush=True)

    proc = subprocess.Popen(
        cmd if isinstance(cmd, list) else cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,  # line-buffered
        env=env,
        cwd=cwd,
    )

    output_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    proc.wait()
    return proc.returncode, "".join(output_lines)


@app.function(
    gpu="A100-80GB:1",
    timeout=3600,  # 60 minutes
    volumes={
        "/models": model_volume,
        "/cache": cache_volume,
    },
)
def test_cuda_inference():
    import subprocess
    import os

    # Print GPU info
    run_streaming("nvidia-smi")
    run_streaming("nvcc --version")

    # One-time cache clear after cuBLAS→cuBLASLt op-set change.
    # Remove this block after first successful run with cuBLASLt.
    if os.path.exists(CACHE_DIR) and os.listdir(CACHE_DIR):
        import shutil
        print(f"\nClearing stale e-graph cache (cuBLASLt migration)...", flush=True)
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)

    # Show cache status
    if os.path.exists(CACHE_DIR):
        cache_files = os.listdir(CACHE_DIR)
        total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
        print(f"\nCache: {len(cache_files)} files ({total_size / 1024 / 1024:.1f} MB)", flush=True)
        for f in sorted(cache_files):
            size = os.path.getsize(os.path.join(CACHE_DIR, f))
            print(f"  {f} ({size / 1024:.1f} KB)", flush=True)
    else:
        print("\nCache: empty (first run)", flush=True)

    # Download model weights using Python API
    print("\nDownloading model weights...", flush=True)
    if not download_model(MAIN_MODEL_REPO, MAIN_MODEL_DIR, ["*.safetensors", "*.json"]):
        return {"status": "download_failed"}

    if not download_model(TOKENIZER_REPO, TOKENIZER_DIR, ["*.safetensors", "*.json", "tokenizer*", "vocab*"]):
        return {"status": "download_failed"}

    model_volume.commit()
    print("Model weights ready.", flush=True)

    # Build the binary
    print("\nBuilding qwen3_tts with CUDA feature...", flush=True)
    rc, _ = run_streaming(
        "cargo build --release -p qwen3_tts --features cuda",
        cwd=MOUNT_PATH,
    )
    if rc != 0:
        return {"status": "build_failed"}

    # Find the built binary
    binary = f"{MOUNT_PATH}/target/release/qwen3_tts_infer"
    if not os.path.exists(binary):
        run_streaming(f"find {MOUNT_PATH}/target/release -name 'qwen3_tts*' -type f")
        return {"status": "binary_not_found"}

    # Run inference with streaming output
    print("\nRunning CUDA inference...", flush=True)
    rc, output = run_streaming(
        binary,
        env={
            **os.environ,
            "QWEN3_TTS_MODEL_DIR": MAIN_MODEL_DIR,
            "QWEN3_TTS_TOKENIZER_DIR": TOKENIZER_DIR,
            "RUST_BACKTRACE": "1",
            "QWEN3_TTS_MAX_FRAMES": "100",
            "LUMINAL_CACHE_DIR": CACHE_DIR,
            "LUMINAL_PROFILE": "1",
        },
    )

    # Commit cache so it persists for next run
    if os.path.exists(CACHE_DIR):
        cache_files = os.listdir(CACHE_DIR)
        total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
        print(f"\nCommitting {len(cache_files)} cache files ({total_size / 1024 / 1024:.1f} MB)...", flush=True)
    cache_volume.commit()

    status = "success" if rc == 0 else "failed"
    print(f"\nInference {status} (exit code: {rc})", flush=True)
    return {"status": status, "returncode": rc}


@app.local_entrypoint()
def main():
    print("Launching CUDA test on Modal A100...")
    result = test_cuda_inference.remote()
    print(f"\nResult: {result['status']}")
