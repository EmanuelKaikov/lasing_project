import pandas as pd
import matplotlib.pyplot as plt

# Load ASTM G173 solar spectrum data
file_path = "C:\\Users\\emanu\\lasing_gpt_version\\ASTMG173.csv"
data = pd.read_csv(file_path, skiprows=1)

# Extract relevant columns and convert units to W/mm²/nm
wavelengths = data['Wvlgth nm']
irradiance_ideal = data['Etr W*m-2*nm-1'] * 1e-6             # Now in W/mm²/nm
irradiance_surface = data['Global tilt  W*m-2*nm-1'] * 1e-6  # Now in W/mm²/nm

# Plot
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, irradiance_ideal, label='Extraterrestrial (Ideal)', color='orange')
plt.plot(wavelengths, irradiance_surface, label='Global Tilt (Earth Surface)', color='blue')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral Irradiance (W/mm²/nm)')
plt.title('Solar Spectral Irradiance: Ideal vs Earth Surface (in W/mm²/nm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
