import numpy as np
import matplotlib.pyplot as plt

try:
    from hcipy import *
except ModuleNotFoundError as e:
    raise ImportError("This script requires the 'hcipy' module. Please install it with 'pip install hcipy' before running.") from e

def simulate_beam_after_8km_only(
    central_wavelength=1e-6,
    bandwidth=10e-9,
    grid_size=512,
    grid_diameter=5,
    beam_waist_input=2e-3,
    z_before_slits=8000.0,
    n_wavelengths=10):

    pupil_grid = make_pupil_grid(grid_size, grid_diameter)
    wavelengths = np.linspace(
        central_wavelength - bandwidth / 2,
        central_wavelength + bandwidth / 2,
        n_wavelengths
    )

    sigma = bandwidth / 2.355
    weights = np.exp(-((wavelengths - central_wavelength) ** 2) / (2 * sigma ** 2))
    weights /= np.sum(weights)

    beam_waist = beam_waist_input if beam_waist_input is not None else grid_diameter / 4
    E_before_slit = np.zeros(pupil_grid.size, dtype=complex)

    for wl, weight in zip(wavelengths, weights):
        field = np.exp(-(pupil_grid.x**2 + pupil_grid.y**2) / beam_waist**2)
        wf = Wavefront(Field(field, pupil_grid), wl)

        prop1 = AngularSpectrumPropagator(pupil_grid, z_before_slits)
        wf = prop1(wf)
        E_before_slit += weight * wf.electric_field

    intensity = np.abs(E_before_slit.reshape((grid_size, grid_size))) ** 2
    intensity = np.real(intensity)

    extent = [-grid_diameter/2 * 1e3, grid_diameter/2 * 1e3]

    # Plot 2D intensity map
    plt.figure(figsize=(6, 5))
    plt.imshow(intensity, extent=extent*2, cmap='Blues')
    plt.title(f"2D Intensity after {z_before_slits} m")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.show()

    # Plot 1D center profile
    center_line = intensity[grid_size // 2, :]
    x_vals = np.linspace(extent[0], extent[1], grid_size)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, center_line)
    plt.title("1D Intensity Profile (center row)")
    plt.xlabel("x (mm)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

simulate_beam_after_8km_only()
