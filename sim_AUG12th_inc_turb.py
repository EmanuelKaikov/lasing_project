import numpy as np
import matplotlib.pyplot as plt

try:
    from hcipy import *
except ModuleNotFoundError as e:
    raise ImportError("This script requires the 'hcipy' module. Please install it with 'pip install hcipy' before running.") from e


# ----------------------- Sunlight helpers (unchanged) -----------------------
def blackbody_spectrum(wavelengths, temperature=5778):
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    spectral_radiance = (2 * h * c**2) / (wavelengths**5) / \
                        (np.exp(h * c / (wavelengths * k * temperature)) - 1)
    return spectral_radiance

def generate_sunlight_field(pupil_grid, n_samples=10, temperature=5778):
    wl_min, wl_max = 0.9e-6, 1.1e-6
    wavelengths = np.linspace(wl_min, wl_max, n_samples)
    spectrum = blackbody_spectrum(wavelengths, temperature)
    spectrum /= np.sum(spectrum)

    total_field = np.zeros(pupil_grid.size, dtype=complex)
    for wl, weight in zip(wavelengths, spectrum):
        rand_phase = np.exp(1j * 2 * np.pi * np.random.rand(pupil_grid.size))
        total_field += weight * rand_phase

    return Field(total_field, pupil_grid)


# ----------------------- Kolmogorov turbulence (built-in + fallback) -----------------------
def _infer_grid_sampling(pupil_grid):
    """Return (N, dx, diameter_m) for a square HCIPy pupil_grid."""
    N = int(round(np.sqrt(pupil_grid.size)))
    x2 = pupil_grid.x.reshape(N, N)
    dx = float(np.mean(np.diff(x2[0, :])))
    diameter = float(x2.max() - x2.min())
    return N, dx, diameter

def _kolmogorov_phase_screen(N, dx, r0):
    """Fourier-domain Kolmogorov phase screen φ(x,y) [rad] on an N×N grid with pixel pitch dx [m]."""
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx, indexing='xy')
    f = np.sqrt(FX**2 + FY**2)
    f[0, 0] = 1e-6  # avoid singularity

    PSD_phi = 0.023 * (r0 ** (-5.0/3.0)) * (f ** (-11.0/3.0))  # rad^2 m^2 / (cycles/m)^2

    cn = (np.random.normal(size=(N, N)) + 1j*np.random.normal(size=(N, N))) / np.sqrt(2.0)

    # Frequency sampling step
    df = 1.0 / (N * dx)

    # Build spectrum; scale by sqrt(PSD)*df (df_x*df_y = df^2 for square grid)
    spectrum = cn * np.sqrt(PSD_phi) * (df)

    phi = np.fft.ifft2(np.fft.ifftshift(spectrum))
    phi = np.real(phi) * (N**2)  # compensate numpy's ifft scaling
    return phi

def propagate_with_turbulence(
    wf,
    pupil_grid,
    wavelength,
    L_total=8000.0,
    N_screens=12,
    r0_total=None,
    Cn2=None,
    L0=25.0,
    l0=0.01,
):
    """
    HCIPy-only split-step turbulence using InfiniteAtmosphericLayer.
    No custom phase screens, no fallbacks.

    Exactly one of r0_total or Cn2 must be provided.
    """
    if (r0_total is None) == (Cn2 is None):
        raise ValueError("Provide exactly one of r0_total or Cn2.")

    k = 2.0 * np.pi / wavelength

    # If user supplied Cn2 for the whole path, convert to total-path r0
    if r0_total is None:
        # r0_total = [0.423 * k^2 * Cn2 * L_total]^(-3/5)
        r0_total = (0.423 * (k**2) * Cn2 * L_total) ** (-3.0/5.0)

    dz = L_total / float(N_screens)

    # Per-step Fried parameter so that N steps reproduce r0_total:
    # r0_step = r0_total * (dz / L_total)^(3/5)
    r0_step = r0_total * (dz / L_total) ** (3.0/5.0)

    # Convert r0_step back to a per-step constant Cn2_step over dz:
    # r0_step = [0.423 * k^2 * Cn2_step * dz]^(-3/5)  =>  Cn2_step = ...
    Cn2_step = (r0_step ** (-5.0/3.0)) / (0.423 * (k**2) * dz)

    step_prop = AngularSpectrumPropagator(pupil_grid, dz)

    # Helper: build a layer across HCIPy versions (constructor signatures differ)
    def make_layer():
        # Try positional (Cn2, wavelength, L0, l0)
        try:
            return InfiniteAtmosphericLayer(pupil_grid, Cn2_step, wavelength, L0, l0)
        except TypeError:
            pass
        # Try positional without outer/inner scales
        try:
            return InfiniteAtmosphericLayer(pupil_grid, Cn2_step, wavelength)
        except TypeError:
            pass
        # Try keyword form
        try:
            return InfiniteAtmosphericLayer(pupil_grid, Cn2=Cn2_step, wavelength=wavelength, L0=L0, l0=l0)
        except TypeError:
            pass
        # Try minimal keyword form
        try:
            return InfiniteAtmosphericLayer(pupil_grid, Cn2=Cn2_step, wavelength=wavelength)
        except Exception as e:
            raise RuntimeError(
                "Could not construct HCIPy InfiniteAtmosphericLayer with any known signature. "
                "Please check your HCIPy version."
            ) from e

    # Apply N layers with free-space propagation between them
    for _ in range(N_screens):
        layer = make_layer()
        # Some versions support __call__, others .forward()
        try:
            wf = layer(wf)
        except TypeError:
            wf = layer.forward(wf)
        wf = step_prop(wf)

    return wf


# ----------------------- Main simulation (your pipeline) -----------------------
def simulate_laser_with_slits(
    central_wavelength=1e-6,
    bandwidth=10e-9,
    grid_size=512,                 # make sure it's int
    grid_diameter=5e-3,            # 5 mm full width
    beam_waist_input=2e-3,         # 2 mm waist at source
    slit_sep=0.5e-3,               # 0.5 mm
    slit_width=1e-4,               # 0.1 mm
    z_before_slits=8000.0,         # fully turbulent segment
    z_after_slits=0.1,             # 10 cm to screen
    n_wavelengths=10,
    # Turbulence controls (choose ONE of the next two)
    r0_total=None,                 # e.g., 0.10 for 10 cm at 1 µm over 8 km
    Cn2=1e-15,                     # OR use Cn2 and compute r0_total for L=8 km
    N_screens=12,
    P=0.5,
    show_plot=True):

    grid_size = int(grid_size)
    pupil_grid = make_pupil_grid(grid_size, grid_diameter)

    wavelengths = np.linspace(
        central_wavelength - bandwidth / 2,
        central_wavelength + bandwidth / 2,
        n_wavelengths
    )
    sigma = bandwidth / 2.355
    weights = np.exp(-((wavelengths - central_wavelength) ** 2) / (2 * sigma ** 2))
    weights /= np.sum(weights)

    x = pupil_grid.x
    slit_mask = ((np.abs(x + slit_sep / 2) < slit_width / 2) |
                 (np.abs(x - slit_sep / 2) < slit_width / 2))

    beam_waist = beam_waist_input if beam_waist_input is not None else grid_diameter / 4

    E_initial = np.zeros(pupil_grid.size, dtype=complex)
    E_before_slit = np.zeros(pupil_grid.size, dtype=complex)
    E_after_slit = np.zeros(pupil_grid.size, dtype=complex)
    E_final = np.zeros(pupil_grid.size, dtype=complex)

    # --- Stage 1: broadband emission + 8 km turbulent propagation ---
    for wl, weight in zip(wavelengths, weights):
        field = np.exp(-(pupil_grid.x**2 + pupil_grid.y**2) / beam_waist**2)

        S = np.sqrt(P / (0.5 * np.pi * beam_waist**2))
        field *= S

        wf = Wavefront(Field(field, pupil_grid), wl)

        # store emitted (summed over spectrum)
        E_initial += weight * wf.electric_field

        # replace vacuum 8 km with turbulence
        wf = propagate_with_turbulence(
            wf, pupil_grid, wavelength=wl,
            L_total=z_before_slits, N_screens=N_screens,
            r0_total=r0_total, Cn2=Cn2
        )
        E_before_slit += weight * wf.electric_field

    # --- Optional: Sunlight added just before slits (kept off by default) ---
    # sunlight = generate_sunlight_field(pupil_grid)
    # sun_attenuation = 0.765  # ASTM G173 at 1050 nm; scale if needed
    # E_before_slit += sun_attenuation * sunlight

    # --- Stage 2: apply slits ---
    E_after_slit = E_before_slit * slit_mask

    # --- Stage 3: short propagation to screen ---
    for wl, weight in zip(wavelengths, weights):
        wf = Wavefront(Field(E_after_slit, pupil_grid), wl)
        prop2 = AngularSpectrumPropagator(pupil_grid, z_after_slits)
        wf = prop2(wf)
        E_final += weight * wf.electric_field

    # --- Utilities & plots ---
    def to_intensity(E):
        I = np.abs(E.reshape((grid_size, grid_size))) ** 2
        return np.real(I)

    def calculate_waist_1e2(E, grid):
        """1/e^2 diameter from the center-line profile (Gaussian-equivalent)."""
        I2 = np.abs(E.reshape((grid_size, grid_size))) ** 2
        x_vals = np.linspace(-grid_diameter/2, grid_diameter/2, grid_size)
        prof = I2[grid_size//2, :]
        I0 = np.max(prof)
        if I0 <= 0:
            return np.nan
        # Find |x| where I = I0 * e^-2
        target = I0 * np.exp(-2)
        # Right side crossing
        idx_max = np.argmax(prof)
        xs = x_vals[idx_max:]
        ys = prof[idx_max:]
        if np.all(ys < target):
            return np.nan
        xr = np.interp(target, ys[::-1], xs[::-1])
        # Diameter = 2*(xr - x_center)
        return 2 * (xr - x_vals[idx_max])

    waist_initial_mm = calculate_waist_1e2(E_initial, pupil_grid) * 1e3
    waist_before_mm  = calculate_waist_1e2(E_before_slit, pupil_grid) * 1e3

    # Theoretical Gaussian beam waist growth (no turbulence)
    z_R = np.pi * beam_waist**2 / central_wavelength
    waist_theory_mm = (beam_waist * np.sqrt(1 + (z_before_slits / z_R)**2)) * 1e3

    intensity_initial = to_intensity(E_initial)
    intensity_before  = to_intensity(E_before_slit)
    intensity_after   = to_intensity(E_after_slit)
    intensity_final   = to_intensity(E_final)

    panels = [
        (intensity_initial, f"Initial Emission\nw(1/e²) ≈ {waist_initial_mm:.2f} mm"),
        (intensity_before,  f"Before Slits ({z_before_slits} m, turbulent)\n"
                            f"w(1/e²) ≈ {waist_before_mm:.2f} mm (Sim)\n"
                            f"{waist_theory_mm:.2f} mm (Vacuum Theory)"),
        (intensity_after,   "After Slits"),
        (intensity_final,   f"Final Pattern {z_after_slits} m after slits")
    ]

    if show_plot:
        extent = [-grid_diameter/2 * 1e3, grid_diameter/2 * 1e3]
        fig, axs = plt.subplots(len(panels), 2, figsize=(12, 3 * len(panels)))
        for i, (I, title) in enumerate(panels):
            ax_img = axs[i, 0]
            ax_prof = axs[i, 1]

            im = ax_img.imshow(I, extent=extent*2, cmap='Blues')
            ax_img.set_title(title)
            ax_img.set_xlabel("x (mm)")
            ax_img.set_ylabel("y (mm)")
            fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, label="Intensity")

            center_line = I[grid_size//2, :]
            ax_prof.plot(np.linspace(extent[0], extent[1], grid_size), center_line)
            ax_prof.set_title("1D Profile across center")
            ax_prof.set_xlabel("x (mm)")
            ax_prof.set_ylabel("Intensity")

        plt.tight_layout()
        plt.show()

        # Also show separate figures per stage (2D + 1D), if you like:
        for I, title in panels:
            plt.figure(figsize=(6, 5))
            plt.imshow(I, extent=extent*2, cmap='Blues')
            plt.title(title)
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.colorbar(label="Intensity")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6, 3))
            center_line = I[grid_size//2, :]
            plt.plot(np.linspace(extent[0], extent[1], grid_size), center_line)
            plt.title(f"1D Profile: {title}")
            plt.xlabel("x (mm)")
            plt.ylabel("Intensity")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return intensity_final


def main():
    # Example: use Cn2 (auto-compute r0_total for 8 km)
    simulate_laser_with_slits(
        central_wavelength=1e-6,
        bandwidth=10e-9,
        grid_size=512,
        grid_diameter=5e-1,
        beam_waist_input=2e-3,
        slit_sep=0.5e-3,
        slit_width=1e-4,
        z_before_slits=100.0,
        z_after_slits=0.1,
        n_wavelengths=8,
        # Choose ONE:
        #r0_total=0.10,        # set total-path r0 directly (meters)
        Cn2=1e-15,              # or provide Cn2 and compute r0 for 8 km
        N_screens=2,
        P = 0.5,
        show_plot=True
    )

if __name__ == '__main__':
    main()
