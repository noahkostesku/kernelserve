# Triton Inference Server Python backend wrapping the cuda-oxide PTX kernel.
#
# Triton calls initialize() once at startup, execute() per batch, finalize() on shutdown.
# See: https://github.com/triton-inference-server/python_backend

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Backend that dispatches inference to the cuda-oxide compiled kernel."""

    def initialize(self, args: dict) -> None:
        """Load model config and set up the kernel wrapper.

        Args:
            args: Dict containing model_config, model_instance_device_id, etc.
        """
        self.model_config = json.loads(args["model_config"])
        # TODO: load the compiled cuda-oxide PTX / shared library
        # TODO: initialize the kernel wrapper (e.g. ctypes or cffi binding)
        # TODO: warm-up pass to JIT-compile on first batch
        self._kernel = None  # placeholder

    def execute(self, requests: list) -> list:
        """Run inference for a batch of requests.

        Args:
            requests: List of pb_utils.InferenceRequest

        Returns:
            List of pb_utils.InferenceResponse (same length as requests)
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            input_np = input_tensor.as_numpy()

            # TODO: copy input to GPU, invoke cuda-oxide kernel, copy output back
            output_np = np.zeros_like(input_np)  # placeholder passthrough

            output_tensor = pb_utils.Tensor("OUTPUT_0", output_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self) -> None:
        """Release GPU resources."""
        # TODO: free CUDA device buffers allocated in initialize()
        pass
