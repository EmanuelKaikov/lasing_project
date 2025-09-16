import numpy as np
import matplotlib.pyplot as plt

try:
    from hcipy import *
except ModuleNotFoundError as e:
    raise ImportError("This script requires the 'hcipy' module. Please install it with 'pip install hcipy' before running.") from e


# --------------------- Turbulence models ---------------------
def kolmogorov_turbulence_layer(pupil_grid, r0=0.1):
    """Kolmogorov turbulence: infinite outer scale (baseline)."""
    return InfiniteAtmosphericLayer(pupil_grid, r0=r0, L0=1e6, l0=0.0)

def von_karman_turbulence_layer(pupil_grid, r0=0.1, L0=25.0, l0=0.01):
    """Von Kármán turbulence: finite outer/inner scales."""
    return InfiniteAtmosphericLayer(pupil_grid, r0=r0, L0=L0, l0=l0)

def multilayer_turbulence(pupil_grid):
    """Multi-layer turbulence with wind advection."""
    altitudes = [0, 1000, 5000, 10000]
    weights   = [0.5, 0.2, 0.2, 0.1]
    wind_speeds = [5, 10, 20, 30]
    wind_dirs   = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    layers = []
    for frac, v, phi in zip(weights, wind_speeds, wind_dirs):
        r0_layer = 0.1 * frac**(-3/5)  # rough scaling
        layers.append(InfiniteAtmosphericLayer(pupil_grid, r0=r0_layer, L0=25.0, l0=0.01,
                                               velocity=v, wind_direction=phi))
    return AtmosphericModel(layers)


# --------------------- Main simulation ---------------------
def simulate_laser_with_slits(
    central_wavelength=1e-6,
    bandwidth=10e-9,
    grid_size=512,
    grid_diameter=5e-3,
    beam_waist_input=2e-3,
    slit_sep=0.5e-3,
    slit_width=1e-4,
    z_before_slits=8000.0,
    z_after_slits=0.1,
    n_wavelengths=10,
    turbulence_mode="none",   # "none" | "kolmogorov" | "von_karman" | "multilayer"
    show_plot=True):

    pupil_grid = make_pupil_grid(grid_size, grid_diameter)
    wavelengths = np.linspace(
        central_wavelength - bandwidth / 2,
        central_wavelength + bandwidth / 2,
        n_wavelengths
    )

    # Gaussian spectral weights
    sigma = bandwidth / 2.355
    weights = np.exp(-((wavelengths - central_wavelength)**2) / (2 * sigma**2))
    weights /= np.sum(weights)

    # Slit mask
    x = pupil_grid.x
    slit_mask = ((np.abs(x + slit_sep/2) < slit_width/2) |
                 (np.abs(x - slit_sep/2) < slit_width/2))

    # Laser setup
    beam_waist = beam_waist_input if beam_waist_input is not None else grid_diameter/4
    E_initial, E_before_slit, E_after_slit, E_final = [np.zeros(pupil_grid.size, dtype=complex) for _ in range(4)]

    # Laser power normalization
    P = 0.5
    norm_factor = np.sqrt(P / (0.5 * np.pi * beam_waist**2))

    # --- Select turbulence model ---
    if turbulence_mode == "kolmogorov":
        turbulence = kolmogorov_turbulence_layer(pupil_grid, r0=0.1)
    elif turbulence_mode == "von_karman":
        turbulence = von_karman_turbulence_layer(pupil_grid, r0=0.1, L0=25.0, l0=0.01)
    elif turbulence_mode == "multilayer":
        turbulence = multilayer_turbulence(pupil_grid)
    else:
        turbulence = None

    # ---------------- Propagation ----------------
    for wl, weight in zip(wavelengths, weights):
        field = np.exp(-(pupil_grid.x**2 + pupil_grid.y**2) / beam_waist**2)
        field *= norm_factor
        wf = Wavefront(Field(field, pupil_grid), wl)
        E_initial += weight * wf.electric_field

        if turbulence is not None:
            if turbulence_mode == "multilayer":
                wf = turbulence(wf, z_before_slits)   # time evolution
            else:
                wf = turbulence(wf)                   # single layer phase screen
        else:
            prop1 = AngularSpectrumPropagator(pupil_grid, z_before_slits)
            wf = prop1(wf)

        E_before_slit += weight * wf.electric_field

    # Slits
    E_after_slit = E_before_slit * slit_mask

    # Final short propagation
    for wl, weight in zip(wavelengths, weights):
        wf = Wavefront(Field(E_after_slit, pupil_grid), wl)
        prop2 = AngularSpectrumPropagator(pupil_grid, z_after_slits)
        wf = prop2(wf)
        E_final += weight * wf.electric_field

    def to_intensity(E):
        return np.abs(E.reshape((grid_size, grid_size)))**2

    intensity_maps = [
        (to_intensity(E_initial), "Initial Emission"),
        (to_intensity(E_before_slit), f"Before Slits ({z_before_slits} m)"),
        (to_intensity(E_after_slit), "After Slits"),
        (to_intensity(E_final), f"Final Pattern {z_after_slits} m after slits")
    ]

    if show_plot:
        extent = [-grid_diameter/2*1e3, grid_diameter/2*1e3]
        for intensity, title in intensity_maps:
            plt.figure(figsize=(6,5))
            plt.imshow(intensity, extent=extent*2, cmap='Blues')
            plt.colorbar(label='Intensity (W/m²)')
            plt.title(title)
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.tight_layout()
        plt.show()

    return intensity_maps[-1][0]


# --------------------- Run ---------------------
if __name__ == "__main__":
    simulate_laser_with_slits(turbulence_mode="none")  # change to "kolmogorov" / "multilayer"
