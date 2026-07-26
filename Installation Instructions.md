1. System Requirements
KV‑Webb is optimized for modern hardware and GPU routing.

Minimum
Rust 1.74+

CUDA‑capable GPU (NVIDIA GTX series or better)

Windows, Linux, or WSL2

Git

Recommended
NVIDIA RTX 30xx / 40xx

CUDA 12.x

Rust nightly (for fastest builds)

32–64 GB RAM

SSD/NVMe storage

2. Clone the Repository
bash
git clone https://github.com/thomaspricetj-hash/kv_webb.git
cd kv_webb
3. Install Dependencies
KV‑Webb uses standard Rust crates plus optional CUDA bindings.

Rust Dependencies
bash
cargo build
Cargo will automatically fetch:

rayon (parallelism)

serde (packet serialization)

ndarray / nalgebra (geometry)

cuda‑sys / cust (if GPU routing enabled)

If you want GPU routing enabled:

CUDA Toolkit
Install CUDA 12.x from NVIDIA’s official site.

Verify CUDA is visible:

bash
nvcc --version
4. Build KV‑Webb Runtime
CPU‑only build
bash
cargo build --release
GPU‑accelerated build
bash
cargo build --release --features gpu
This enables:

Hybrid PKM GPU routing

Multi‑stream CUDA execution

GPU mask building

GPU‑accelerated heatmaps

GPU‑accelerated pruning

5. Run KV‑Webb Runtime
Default runtime
bash
cargo run --release
GPU runtime
bash
cargo run --release --features gpu
6. Optional: Enable Debug Metrics
bash
cargo run --release --features debug
This prints:

heatmap layers

drift physics

pruning decisions

predictor activity

SSL firewall events

roundabout routing traces

7. Optional: Enable BitDrop_v2 Compression
bash
cargo run --release --features compression
This activates:

reversible cognitive packets

BD3D folding

collapse ordering

GPU‑ready packet streaming

8. Optional: Enable Full Cognitive Stack
Everything enabled:

bash
cargo run --release --features "gpu compression debug ssl predictor roundabout"
This runs KV‑Webb in full Tier‑8 cognitive substrate mode.

9. Environment Variables (Optional)
GPU routing tuning
bash
set KVWEB_GPU_STREAMS=8
set KVWEB_GPU_BLOCK_SIZE=256
Predictor tuning
bash
set KVWEB_PREDICTOR_DECAY=0.92
SSL hardening
bash
set KVWEB_SSL_MODE=aggressive
10. Verify Installation
Run the built‑in diagnostic:

bash
cargo test
You should see:

semantic geometry tests

heatmap layer tests

pruning physics tests

drift physics tests

SSL firewall tests

predictor foresight tests

roundabout routing tests

11. No‑Loophole Protection
Add this to your README or LICENSE:

KV‑Webb Runtime 3.4.1 and all associated cognitive routing, geometry, predictor, compression, and security systems are proprietary intellectual property belonging exclusively to Thomas.
No reuse, replication, modification, redistribution, or derivative works are permitted without explicit written permission.
All rights reserved. No exceptions.
