# AlexNet Source Code Overview

## Repository layout
- **`src/`** — CUDA/C++ implementation of the neural network engine, including `convnet.cu` for the multi-GPU manager, `pyconvnet.cu` for the Python extension module, and individual files for layers, costs, data movement, and utilities.
- **Python front end** — Scripts such as `convnet.py`, `layer.py`, and the `convdata*.py` providers implement experiment configuration, layer parsing, and data input pipelines.
- **Support scripts** — Utilities for building (`build.sh`, `package.sh`), running experiments (`test.sh`, `run4.sh`), inspecting networks (`shownet.py`), and manipulating datasets.

## Execution flow
1. **Configuration parsing**: Python reads layer definitions via `LayerParser` classes and prepares the model state (`layer.py`).
2. **Model initialization**: The Python `ConvNet` class imports the compiled CUDA extension and sends parsed layer parameters and hardware configuration to `initModel` (`convnet.py`).
3. **GPU orchestration**: The C++ `ConvNet` manager instantiates data layers on the host, spawns `ConvNetGPU` worker threads per device, connects the layer graph, and schedules asynchronous work items (`src/convnet.cu`).
4. **Asynchronous training**: Python data providers supply batches; the extension enqueues `TrainingWorker`/`MultiviewTestWorker` instances that run forward/backward passes on GPUs and return results via queues (`src/pyconvnet.cu`).

## Division of labor: Python vs. CUDA
- The **Python front end** is primarily orchestration: it parses configuration files, sets experiment options, and streams minibatches. Once a batch is ready, Python calls into the extension and waits for completion.
- The **CUDA/C++ backend** performs the heavy lifting: GPU kernels in `src/layer_kernels.cu`, `weights.cu`, `cost.cu`, and related files execute convolutions, activations, pooling, weight updates, and loss calculations. Worker threads coordinate these kernels, handle synchronization, and manage GPU memory.
- Host-side C++ utilities handle queueing, synchronization primitives, and statistics aggregation so the GPUs stay busy with compute-intensive work while Python remains a lightweight controller.

## Key subsystems
- **Layer and neuron parsing**: `layer.py` contains parser classes that map configuration strings to layer objects, including support for parameterized neurons and weight sharing logic.
- **Data pipeline**: `convdata.py` and related modules implement ImageNet and CIFAR data providers, mean subtraction, cropping, color PCA noise, and multiview testing support.
- **CUDA kernels**: Files like `layer_kernels.cu`, `cost.cu`, and `weights.cu` hold the computational kernels used by GPU threads.

## Next steps for newcomers
- Build the CUDA extension using `build.sh` and explore `Makefile-distrib` to understand compilation targets.
- Review sample experiment configurations in `example-layers/` to see how layer definitions map to parser logic.
- Run utility scripts like `shownet.py` to visualize activations or `test.py` for regression tests.
- Dive into the multi-GPU scheduling code in `src/worker.cu` and `src/util.cu` to understand the threading model.

## How the 2012 code holds up in 2025
- **Environment expectations are archaic**: the build scripts target CUDA 4.x/5.0-era toolchains, Python 2.7, and Kepler GPUs. Running it today means containerizing an old toolchain or heavily patching the build system.
- **No vendor libraries**: everything from convolutions to pooling is hand-rolled. That was necessary pre-cuDNN, but by 2025 you would lean on cuDNN, CUTLASS, or PyTorch primitives unless you need custom research kernels.
- **Manual threading and memory management**: the worker queues, mutexes, and raw GPU allocations in `src/` demand intimate knowledge of the engine. Modern frameworks abstract this behind graph compilers, autograd, and unified memory helpers.
- **Minimal safety net**: there are no unit tests, limited assertions, and the Python front end lacks type hints or linters. Debugging requires printf-style logging and GPU asserts.
- **Tightly coupled design**: layer definitions, scheduling, and kernel code are interwoven, making it hard to swap components or experiment with novel layers compared to modular 2025 frameworks.
- **Historical value remains high**: despite the rough edges, the repository is a valuable case study in early large-scale GPU training, especially for understanding how AlexNet squeezed performance from hardware of the era.

### Modernization pointers
- Containerize the legacy toolchain (Python 2.7, CUDA 5.0) or port the build to a contemporary compiler with updated CUDA APIs.
- Replace bespoke kernels with cuDNN/CUTLASS equivalents where possible, keeping only genuinely novel kernels.
- Introduce automated tests around the Python API and unit tests for kernels using modern CUDA testing harnesses.
- Layer the codebase behind clearer interfaces (e.g., C API or pybind11 bindings) so experiment code can be written in modern Python 3.
