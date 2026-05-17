use pyo3::prelude::*;

/// Run RMS-Norm via the cuda-oxide GPU kernel (--features gpu) or the CPU
/// reference implementation (default, no CUDA required).
///
/// Args:
///     input:  Flat f32 list of shape [batch * hidden_dim].
///     weight: f32 list of shape [hidden_dim].
///     eps:    Numerical stability epsilon (default 1e-5).
///
/// Returns:
///     Flat f32 list of the same length as `input`.
#[pyfunction]
#[pyo3(signature = (input, weight, eps = 1e-5))]
fn rms_norm(input: Vec<f32>, weight: Vec<f32>, eps: f32) -> PyResult<Vec<f32>> {
    #[cfg(feature = "gpu")]
    {
        use kernelserve_kernels::RmsNorm;
        use pyo3::exceptions::PyRuntimeError;
        return RmsNorm::new(eps)
            .forward(&input, &weight)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }

    #[cfg(not(feature = "gpu"))]
    {
        use kernelserve_kernels::cpu_ref::rms_norm_cpu;
        let hidden_dim = weight.len();
        let batch = if hidden_dim == 0 { 0 } else { input.len() / hidden_dim };
        Ok(rms_norm_cpu(&input, &weight, batch, hidden_dim, eps))
    }
}

// Function name must match [tool.maturin] module-name = "kernelserve_core"
#[pymodule]
fn kernelserve_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rms_norm, m)?)?;
    Ok(())
}
