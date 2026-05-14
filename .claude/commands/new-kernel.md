---
description: Scaffold a new kernel with all required boilerplate across cuda_oxide, Triton, serving, and tests
argument-hint: [kernel-name]
allowed-tools: [Bash, Read, Write, Edit]
---

# /new-kernel — Kernel Scaffolder

## Steps

1. If $ARGUMENTS is empty, ask: "Kernel name? (snake_case, e.g. rms_norm)"
   Validate it is snake_case with no hyphens and is a valid Rust identifier.
   Ask: "One-sentence description of what this kernel computes?"

2. Check the name does not already exist:
   ```bash
   ls kernels/cuda_oxide/src/<name>.rs 2>/dev/null && echo "EXISTS"
   ```
   Abort if it already exists.

3. Create `kernels/cuda_oxide/src/<name>.rs`:
   ```rust
   use cuda_oxide::prelude::*;

   #[kernel]
   pub fn <name>(/* TODO: add parameters */) {
       todo!("<description>");
   }
   ```

4. Create `kernels/triton/<name>.py`:
   ```python
   import triton
   import triton.language as tl
   import torch

   @triton.jit
   def <name>_kernel(
       # TODO: add pointer and stride arguments
       BLOCK_SIZE: tl.constexpr,
   ):
       pass  # TODO: implement

   def <name>(x: torch.Tensor) -> torch.Tensor:
       """<description>"""
       raise NotImplementedError
   ```

5. Create `tests/unit/test_<name>.py`:
   ```python
   import pytest
   import torch

   @pytest.mark.parametrize("shape", [(128, 512), (256, 1024)])
   def test_<name>_correctness(shape):
       x = torch.randn(*shape, device="cpu")
       ref = None   # TODO: PyTorch reference output
       out = None   # TODO: call kernel under test
       assert torch.allclose(ref, out, atol=1e-4), f"max err {(ref - out).abs().max()}"
   ```

6. Create `serving/triton_backends/<name>/config.pbtxt`:
   ```
   name: "<name>"
   backend: "python"
   max_batch_size: 32

   input [{ name: "INPUT" data_type: TYPE_FP32 dims: [-1, -1] }]
   output [{ name: "OUTPUT" data_type: TYPE_FP32 dims: [-1, -1] }]

   instance_group [{ kind: KIND_GPU count: 1 }]
   ```

7. Register in `kernels/cuda_oxide/src/lib.rs` — add after the last `pub mod` line:
   ```rust
   pub mod <name>;
   pub use <name>::*;
   ```

8. Run a quick compile check:
   ```bash
   cd kernels/cuda_oxide && cargo check 2>&1 | tail -5
   ```
   And verify test discovery:
   ```bash
   pytest tests/unit/test_<name>.py --collect-only -q
   ```

9. Show all created file paths. Ask: "Commit scaffolding now? (y/n)"
   If yes, commit with exactly: `feat: scaffold <name> kernel`
