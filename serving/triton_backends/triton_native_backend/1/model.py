# Triton Inference Server Python backend wrapping the Triton (GPU language) kernel baseline.
#
# Dispatches inference to kernels/triton/ implementations for benchmarking comparison
# against the cuda-oxide backend.

import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Backend that dispatches inference to the Python/Triton kernel baseline."""

    def initialize(self, args: dict) -> None:
        self.model_config = json.loads(args["model_config"])
        # TODO: determine which kernel to use based on model config parameters
        # TODO: import and warm up the relevant kernels/triton/<kernel>.py function
        self._device = f"cuda:{args.get('model_instance_device_id', 0)}"
        self._kernel_fn = None  # placeholder

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            input_np = input_tensor.as_numpy()

            # TODO: convert input_np to torch tensor on self._device
            # TODO: call self._kernel_fn(input_tensor_cuda, ...)
            # TODO: move result back to CPU as numpy
            output_np = np.zeros_like(input_np)  # placeholder passthrough

            output_tensor = pb_utils.Tensor("OUTPUT_0", output_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self) -> None:
        pass
