import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

np.random.seed(42)
# Parameters
wards = ['A', 'B', 'C', 'D', 'E']
arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0}
mean_stay = {'A': 2.9, 'B': 4.0, 'C': 4.5, 'D': 1.4, 'E': 3.9}
capacities = {'A': 55, 'B': 40, 'C': 30, 'D': 20, 'E': 20}
relocation_probs = {
    'A': [0.00, 0.05, 0.10, 0.05, 0.80],
    'B': [0.20, 0.00, 0.50, 0.15, 0.15],
    'C': [0.30, 0.20, 0.00, 0.20, 0.30],
    'D': [0.35, 0.30, 0.05, 0.00, 0.30],
    'E': [0.20, 0.10, 0.60, 0.10, 0.00]
}

# Verify that relocation probabilities sum to 1 for each patient type
for key, probs in relocation_probs.items():
    assert np.isclose(sum(probs), 1.0), f"Probabilities for {key} do not sum to 1"

# Initialize the state of the system
ward_occupancy = {ward: 0 for ward in wards}

# Function to run the simulation
def simulate_hospital(days):
    total_admissions = {ward: 0 for ward in wards}
    total_relocations = {ward: 0 for ward in wards}
    total_losses = {ward: 0 for ward in wards}
    
    for day in range(days):
        for ward in wards:
            arrivals = np.random.poisson(arrival_rates[ward])
            for _ in range(arrivals):
                if ward_occupancy[ward] < capacities[ward]:
                    ward_occupancy[ward] += 1
                    total_admissions[ward] += 1
                else:
                    relocated = False
                    for j, prob in enumerate(relocation_probs[ward]):
                        if np.random.rand() < prob:
                            alt_ward = wards[j]
                            if ward_occupancy[alt_ward] < capacities[alt_ward]:
                                ward_occupancy[alt_ward] += 1
                                total_relocations[alt_ward] += 1
                                relocated = True
                                break
                    if not relocated:
                        total_losses[ward] += 1

            # Handle patient departures based on length of stay
            departures = np.random.poisson(ward_occupancy[ward] / mean_stay[ward])
            ward_occupancy[ward] = max(0, ward_occupancy[ward] - departures)
    
    return total_admissions, total_relocations, total_losses, ward_occupancy

# Example simulation for 365 days
total_admissions, total_relocations, total_losses, final_occupancy = simulate_hospital(365)

# Display results
results = pd.DataFrame({
    'Admissions': total_admissions,
    'Relocations': total_relocations,
    'Losses': total_losses,
    'Final Occupancy': final_occupancy
})
print(results)

# Plotting patient admissions in each ward
plt.figure(figsize=(10, 6))
plt.bar(results.index, results['Losses'], color='skyblue')
plt.xlabel('Ward')
plt.ylabel('Number of Losses')
plt.title('Patient Losses in Each Ward Over One Year')
plt.savefig('patient_LossesTask1.png')
plt.show()
