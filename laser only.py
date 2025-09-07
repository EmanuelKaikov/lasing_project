import numpy as np
import matplotlib.pyplot as plt

try:
    from hcipy import *
except ModuleNotFoundError as e:
    raise ImportError("This script requires the 'hcipy' module. Please install it with 'pip install hcipy' before running.") from e

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

def simulate_laser_with_slits(
    central_wavelength=1e-6,
    bandwidth=10e-9,
    grid_size=4096,
    grid_diameter=5,
    beam_waist_input=2e-3,
    slit_sep=0.5e-3,
    slit_width=1e-4,
    z_before_slits=8000.0,
    z_after_slits=0.1,
    n_wavelengths=10,
    show_plot=True):

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

    # Desired source power (Watts)
    P = 0.5

    for wl, weight in zip(wavelengths, weights):
        # Initial Gaussian field (unnormalized)
        field = np.exp(-(pupil_grid.x**2 + pupil_grid.y**2) / beam_waist**2)

        # Power normalization to P watts for TEM00:
        # P = S^2 * (pi * w0^2 / 2)  =>  S = sqrt(P / (0.5*pi*w0^2))
        S = np.sqrt(P / (0.5 * np.pi * beam_waist**2))
        field *= S

        wf = Wavefront(Field(field, pupil_grid), wl)
        E_initial += weight * wf.electric_field

        # 8 km free-space propagation (vacuum)
        prop1 = AngularSpectrumPropagator(pupil_grid, z_before_slits)
        wf = prop1(wf)
        E_before_slit += weight * wf.electric_field

    """# Add sunlight (scaled small relative to laser)
    sunlight = generate_sunlight_field(pupil_grid)
    sun_scaling = 0.765  # ASTM-based attenuation near 1050 nm
    #E_before_slit += sun_scaling * sunlight"""

    # Apply double slit
    E_after_slit = E_before_slit * slit_mask

    # Short propagation after slits
    for wl, weight in zip(wavelengths, weights):
        wf = Wavefront(Field(E_after_slit, pupil_grid), wl)
        prop2 = AngularSpectrumPropagator(pupil_grid, z_after_slits)
        wf = prop2(wf)
        E_final += weight * wf.electric_field

    def to_intensity(E):
        intensity = np.abs(E.reshape((grid_size, grid_size))) ** 2
        return np.real(intensity)

    def calculate_waist(E, grid):
        intensity = np.abs(E)**2
        x = grid.x.reshape((grid_size, grid_size))
        x_vals = np.linspace(-grid_diameter/2, grid_diameter/2, grid_size)
        I = intensity.reshape((grid_size, grid_size))
        x_profile = np.sum(I, axis=0)
        I_norm = x_profile / np.trapezoid(x_profile, x_vals)
        x2_mean = np.trapezoid(x_vals**2 * I_norm, x_vals)
        waist = 2 * np.sqrt(x2_mean)
        return waist

    waist_initial = calculate_waist(E_initial, pupil_grid) * 1e3
    waist_before = calculate_waist(E_before_slit, pupil_grid) * 1e3
    z_R = np.pi * beam_waist**2 / central_wavelength
    waist_theoretical = beam_waist * np.sqrt(1 + (z_before_slits / z_R)**2) * 1e3

    intensity_initial = to_intensity(E_initial)
    intensity_before  = to_intensity(E_before_slit)
    intensity_after   = to_intensity(E_after_slit)
    intensity_final   = to_intensity(E_final)

    intensity_maps = [
        (intensity_initial, f"Initial Emission"),
        (intensity_before,  f"Before Slits"),
        (intensity_after,   "After Slits"),
        (intensity_final,   f"Final Pattern {z_after_slits} meters after slits")
    ]

    if show_plot:
        extent = [-grid_diameter/2 * 1e3, grid_diameter/2 * 1e3]

        # Separate plots for each stage (2D + 1D)
        for intensity, title in intensity_maps:
            # 2D intensity map
            plt.figure(figsize=(6, 5))
            plt.imshow(intensity, extent=extent*2, cmap='Blues')
            plt.colorbar(label='Intensity [W/m^2]')
            plt.title(title)
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.grid(False)
            plt.tight_layout()

            # 1D profile (horizontal center line)
            plt.figure(figsize=(6, 3))
            center_line = intensity[grid_size//2, :]
            plt.plot(np.linspace(extent[0], extent[1], grid_size), center_line)
            plt.title(f"1D Profile: {title}")
            plt.xlabel("x (mm)")
            plt.ylabel("Intensity [W/m^2]")
            plt.grid(True)
            plt.tight_layout()

        plt.show()

    return intensity_maps[-1][0]


def main():
    simulate_laser_with_slits(show_plot=True)

if __name__ == '__main__':
    main()
