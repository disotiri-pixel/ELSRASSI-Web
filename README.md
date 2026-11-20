# ELSRASSI Web

Browser-based 1D linear / equivalent-linear site-response analysis for vertically propagating SH waves. The numerical core is a Python port of the revised MATLAB ELSRASSI formulation benchmarked against STRATA in this project.

## Features

- Acceleration upload as one column (acceleration) or two columns (time, acceleration).
- Input units: cm/s², m/s², or g; arbitrary scale factor.
- Input motion interpreted as **outcrop bedrock** or **within bedrock**.
- Editable or CSV-uploaded layered Vs/density profile.
- Linear analysis with fixed layer damping.
- Equivalent-linear analysis with convergence tolerance, maximum iterations, effective-strain ratio and under-relaxation.
- Darendeli curves using internally computed mean effective confining stress, with optional groundwater.
- Custom modulus-reduction / damping curves assigned by `curve_id`.
- Interactive plots for time histories, transfer function, Fourier spectra, Vs, PGA, shear stress, shear strain, damping and G/Gmax.
- ZIP export of numerical results.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy online with Streamlit Community Cloud

1. Open Streamlit Community Cloud and create a new app.
2. Select repository `disotiri-pixel/ELSRASSI-Web`.
3. Select branch `main`.
4. Select `app.py` as the entry point.
5. Deploy. No secrets or external services are required.

## Motion file format

Either:

```text
0.000   12.4
0.005   18.7
0.010   21.1
```

or a single acceleration column. For one-column input, set `dt` in the UI.

## Profile CSV format

Required columns:

- `thickness_m`
- `Vs_mps`
- `density_Mg_m3`

Optional columns:

- `damping_percent` (used by linear analysis)
- `PI`, `OCR`, `K0` (used when layer-specific Darendeli parameters are enabled)
- `curve_id` (used by custom curves)

The profile table contains **soil layers only**. Bedrock properties are entered separately in the UI.

## Custom curve CSV format

```text
curve_id,strain_percent,G_over_Gmax,damping_percent
A,0.0001,1.0,0.24
A,0.001,1.0,0.8
A,0.01,0.941,2.8
...
```

Strain is in percent. The app uses linear interpolation and clamps outside the supplied strain range to the nearest endpoint, matching the revised MATLAB custom-curve behavior.

## Numerical conventions

- Internal acceleration is converted to cm/s² to retain the MATLAB shear-strain convention.
- `Outcrop bedrock` uses the synthetic outcrop level (MATLAB `N+1`).
- `Within bedrock` uses the top-of-bedrock level (MATLAB `N`).
- Mean effective confining stress for Darendeli is `sigma_m' = sigma_v' (1 + 2 K0) / 3`.
- The final response is recomputed with the final compatible properties after equivalent-linear iteration.

## Important

This app is intended for engineering/research workflows and should be independently checked for project-specific design use.
