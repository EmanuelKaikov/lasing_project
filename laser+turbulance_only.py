import numpy as np
import matplotlib.pyplot as plt

# Requires HCIPy
try:
    from hcipy import *
except ModuleNotFoundError as e:
    raise ImportError("This script requires the 'hcipy' module. Install with: pip install hcipy") from e


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
    Propagate a wavefront through turbulence using HCIPy InfiniteAtmosphericLayer only.
    Works across different HCIPy versions.
    """
    if (r0_total is None) == (Cn2 is None):
        raise ValueError("Provide exactly one of r0_total or Cn2.")

    k = 2 * np.pi / wavelength
    if Cn2 is None:
        # derive constant Cn2 from r0_total
        Cn2 = (r0_total ** (-5.0/3.0)) / (0.423 * (k**2) * L_total)

    dz = L_total / float(N_screens)
    step_prop = AngularSpectrumPropagator(pupil_grid, dz)

    for _ in range(N_screens):
        # Try with keywords, then fallback to positional args
        try:
            layer = InfiniteAtmosphericLayer(pupil_grid, Cn2, wavelength,
                                             L0=L0, l0=l0)
        except TypeError:
            layer = InfiniteAtmosphericLayer(pupil_grid, Cn2, wavelength,
                                             L0, l0)

        # Apply the layer
        try:
            wf = layer(wf)
        except TypeError:
            wf = layer.forward(wf)

        wf = step_prop(wf)

    return wf


# ----------------------- Main simulation (pipeline) -----------------------
def simulate_laser_with_slits(
    central_wavelength=1e-6,
    bandwidth=10e-9,
    grid_size=512,                 # ensure int
    grid_diameter=5e-3,            # 5 mm full width
    beam_waist_input=2e-3,         # 2 mm waist at source
    slit_sep=0.5e-3,               # 0.5 mm
    slit_width=1e-4,               # 0.1 mm
    z_before_slits=8000.0,         # fully turbulent segment
    z_after_slits=0.1,             # 10 cm to screen
    n_wavelengths=10,
    # Turbulence controls (choose ONE of the next two)
    r0_total=None,                 # e.g. 0.10 for 10 cm at 1 µm over 8 km
    Cn2=1e-15,                     # OR use Cn2 and compute r0_total for L=8 km
    N_screens=12,
    P=0.5,
    show_plot=True
):
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

    # --- Stage 1: broadband emission + turbulent propagation ---
    for wl, weight in zip(wavelengths, weights):
        field = np.exp(-(pupil_grid.x**2 + pupil_grid.y**2) / beam_waist**2)

        # Power-normalize Gaussian so total optical power ≈ P
        S = np.sqrt(P / (0.5 * np.pi * beam_waist**2))
        field *= S

        wf = Wavefront(Field(field, pupil_grid), wl)

        # store emitted (summed over spectrum)
        E_initial += weight * wf.electric_field

        # replace vacuum with HCIPy turbulence
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
        cy = grid_size // 2
        profile = I2[cy, :]
        x = pupil_grid.x.reshape(grid_size, grid_size)[cy, :]
        Imax = np.max(profile)
        target = Imax / np.e**2
        # find width at 1/e^2 (simple nearest neighbors)
        idx = np.where(profile >= target)[0]
        if len(idx) < 2:
            return np.nan
        return float(x[idx[-1]] - x[idx[0]])

    I_initial = to_intensity(E_initial)
    I_before = to_intensity(E_before_slit)
    I_after = to_intensity(E_after_slit)
    I_final = to_intensity(E_final)

    if show_plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        im0 = axs[0, 0].imshow(I_initial, origin='lower', extent=[-grid_diameter/2, grid_diameter/2, -grid_diameter/2, grid_diameter/2])
        axs[0, 0].set_title('Initial intensity')
        plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        im1 = axs[0, 1].imshow(I_before, origin='lower', extent=[-grid_diameter/2, grid_diameter/2, -grid_diameter/2, grid_diameter/2])
        axs[0, 1].set_title('Before slits (after turbulence)')
        plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

        im2 = axs[1, 0].imshow(I_after, origin='lower', extent=[-grid_diameter/2, grid_diameter/2, -grid_diameter/2, grid_diameter/2])
        axs[1, 0].set_title('After slits')
        plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

        im3 = axs[1, 1].imshow(I_final, origin='lower', extent=[-grid_diameter/2, grid_diameter/2, -grid_diameter/2, grid_diameter/2])
        axs[1, 1].set_title('Screen (z = %.3f m)' % z_after_slits)
        plt.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    # Diagnostics (optional)
    waist_1e2_initial = calculate_waist_1e2(E_initial, pupil_grid)
    waist_1e2_before = calculate_waist_1e2(E_before_slit, pupil_grid)

    return I_final


def main():
    _ = simulate_laser_with_slits(
        central_wavelength=1.0e-6,
        bandwidth=10e-9,
        grid_size=512,
        grid_diameter=5e-3,
        beam_waist_input=2e-3,
        slit_sep=0.5e-3,
        slit_width=1e-4,
        z_before_slits=8000.0,
        z_after_slits=0.1,
        n_wavelengths=10,
        # Choose one turbulence parameterization:
        # r0_total=0.10,
        Cn2=1e-15,
        N_screens=2,
        P=0.5,
        show_plot=True
    )

if __name__ == "__main__":
    main()
