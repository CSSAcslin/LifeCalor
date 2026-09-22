# LifeCalor 1.1.1 Stage A validation

This record covers the scientific contracts and CUDA prototypes introduced
before CWT or lifetime GPU execution is exposed in the application.

## CPU reference

Run from the project root:

~~~powershell
H:\Newera\programing\.venv\Scripts\python.exe -m unittest tests.test_compute_111_stage_a
~~~

The reference suite verifies:

- all seven CWT wavelets currently offered by the UI;
- PyWavelets integral-kernel sampling, convolution, differentiation and crop;
- CWT wavelet-name normalization without silently accepting new wavelets;
- single- and double-exponential analytic Jacobians;
- the existing SciPy TRF bounds and float64 lifetime contract;
- feasible double-exponential initial values;
- named double-exponential map outputs and fit-status values.

## Target NVIDIA validation

The CUDA prototypes are intentionally not registered as application
capabilities yet. On the target NVIDIA computer, run:

~~~powershell
$env:LIFECALOR_RUN_CUDA_TESTS = "1"
H:\Newera\programing\.venv\Scripts\python.exe -m unittest tests.test_compute_111_cuda_prototypes -v
~~~

Record the GPU name, driver, CUDA runtime, CuPy version, maximum absolute and
relative CWT error, and whether both lifetime Jacobian cases pass. A skipped
test, CPU fallback, or successful CUDA self-test alone is not GPU algorithm
acceptance.

## Current boundary

- CWT CUDA performs the PyWavelets-compatible per-scale convolution,
  differentiation, crop and immediate scale reduction, but is not connected to
  the worker or UI until target-device validation succeeds.
- Lifetime CUDA currently validates model and analytic-Jacobian evaluation.
  The bounded batch solver, convergence diagnostics and CPU-review policy
  remain required before the lifetime GPU capability can be registered.
- Double-exponential map CPU reference returns tau1_map, tau2_map,
  amplitude1_map, amplitude2_map, baseline_map, r_squared_map and fit_status.
  The user-facing result router remains disabled until display, history
  selection and export preserve every field.
