import numpy as np
import pandas as pd
from scipy.stats import lognorm


import random
np.random.seed(42)

# Parameters
wards = ['A', 'B', 'C', 'D', 'E', 'F']
arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0, 'F': 13.0}
mean_stay = {'A': 2.9, 'B': 4.0, 'C': 4.5, 'D': 1.4, 'E': 3.9, 'F': 2.2}
initial_capacities = {'A': 55, 'B': 40, 'C': 30, 'D': 20, 'E': 20, 'F': 27}
urgency_points = {'A': 7, 'B': 5, 'C': 2, 'D': 10, 'E': 5}
relocation_probs = {
    'A': [0.00, 0.05, 0.10, 0.05, 0.80],
    'B': [0.20, 0.00, 0.50, 0.15, 0.15],
    'C': [0.30, 0.20, 0.00, 0.20, 0.30],
    'D': [0.35, 0.30, 0.05, 0.00, 0.30],
    'E': [0.20, 0.10, 0.60, 0.10, 0.00]
}

def update_relocation_probs(relocation_probs):
    for ward in relocation_probs:
        relocation_probs[ward].append(0.0 if ward != 'F' else 1.0)
    relocation_probs['F'] = [0.20, 0.20, 0.20, 0.20, 0.20, 0.0]
    return relocation_probs

def simulate_hospital_with_lognorm(days, adjusted_capacities, relocation_probs, variances):
    ward_occupancy = {ward: 0 for ward in adjusted_capacities}
    total_admissions = {ward: 0 for ward in adjusted_capacities}
    total_relocations = {ward: 0 for ward in adjusted_capacities}
    total_losses = {ward: 0 for ward in adjusted_capacities}
    total_occupied_on_arrival = {ward: 0 for ward in adjusted_capacities}
    
    for day in range(days):
        for ward in adjusted_capacities:
            arrivals = np.random.poisson(arrival_rates[ward])
            for _ in range(arrivals):
                if ward_occupancy[ward] < adjusted_capacities[ward]:
                    ward_occupancy[ward] += 1
                    total_admissions[ward] += 1
                else:
                    total_occupied_on_arrival[ward] += 1
                    relocated = False
                    for j, prob in enumerate(relocation_probs[ward]):
                        if np.random.rand() < prob:
                            alt_ward = list(adjusted_capacities.keys())[j]
                            if ward_occupancy[alt_ward] < adjusted_capacities[alt_ward]:
                                ward_occupancy[alt_ward] += 1
                                total_relocations[alt_ward] += 1
                                relocated = True
                                break
                    if not relocated:
                        total_losses[ward] += 1

            mean = np.log(mean_stay[ward]**2 / np.sqrt(variances[ward] + mean_stay[ward]**2))
            sigma = np.sqrt(np.log(variances[ward] / mean_stay[ward]**2 + 1))
            departures = np.random.poisson(ward_occupancy[ward] / lognorm.rvs(sigma, scale=np.exp(mean), size=1))
            ward_occupancy[ward] = max(0, ward_occupancy[ward] - departures)
    
    prob_all_beds_occupied = {ward: total_occupied_on_arrival[ward] / (total_admissions[ward] + total_occupied_on_arrival[ward]) 
                              for ward in adjusted_capacities}
    expected_admissions = {ward: total_admissions[ward] / days for ward in adjusted_capacities}
    expected_relocations = {ward: total_relocations[ward] / days for ward in adjusted_capacities}
    
    return total_admissions, total_relocations, total_losses, ward_occupancy, prob_all_beds_occupied, expected_admissions, expected_relocations

def format_results(title, results):
    total_admissions, total_relocations, total_losses, final_occupancy, prob_all_beds_occupied, expected_admissions, expected_relocations = results

    print(f"\n{title}\n")
    print(f"{'Ward':<5} {'Admissions':>12} {'Relocations':>12} {'Losses':>8} {'Occupancy':>10} {'Prob Full':>10} {'Exp Admissions':>15} {'Exp Relocations':>17}")
    print("-" * 90)
    for ward in wards:
        occupancy = final_occupancy[ward][0] if isinstance(final_occupancy[ward], np.ndarray) else final_occupancy[ward]
        print(f"{ward:<5} {total_admissions[ward]:>12} {total_relocations[ward]:>12} {total_losses[ward]:>8} {occupancy:>10} {prob_all_beds_occupied[ward]:>10.3f} {expected_admissions[ward]:>15.3f} {expected_relocations[ward]:>17.3f}")

# Different bed distribution scenarios
# Scenario 1: Increase beds in high-urgency wards (e.g., Ward D)
capacities_scenario1 = initial_capacities.copy()
capacities_scenario1['D'] += 10  # Increase beds in Ward D by 10
capacities_scenario1['A'] -= 3
capacities_scenario1['B'] -= 3
capacities_scenario1['C'] -= 2
capacities_scenario1['E'] -= 2
capacities_scenario1['F'] = 27  # Keep Ward F constant

# Scenario 2: Even distribution of beds
total_beds = sum(initial_capacities.values())
even_distribution = total_beds // len(wards)
capacities_scenario2 = {ward: even_distribution for ward in wards}

# Scenario 3: Increase beds in high-arrival wards (e.g., Wards A and F)
capacities_scenario3 = initial_capacities.copy()
capacities_scenario3['A'] += 10  # Increase beds in Ward A by 10
capacities_scenario3['F'] += 10  # Increase beds in Ward F by 10
capacities_scenario3['B'] -= 5
capacities_scenario3['C'] -= 5
capacities_scenario3['D'] -= 5
capacities_scenario3['E'] -= 5

# Set variances for log-normal distribution
variances = {ward: 2 / (mean_stay[ward] ** 2) for ward in wards}

# Run simulations for each scenario
relocation_probs_updated = update_relocation_probs(relocation_probs.copy())
results_scenario1 = simulate_hospital_with_lognorm(365, capacities_scenario1, relocation_probs_updated, variances)
results_scenario2 = simulate_hospital_with_lognorm(365, capacities_scenario2, relocation_probs_updated, variances)
results_scenario3 = simulate_hospital_with_lognorm(365, capacities_scenario3, relocation_probs_updated, variances)

# Display results for each scenario
format_results("Scenario 1: Increase beds in high-urgency wards", results_scenario1)
format_results("Scenario 2: Even distribution of beds", results_scenario2)
format_results("Scenario 3: Increase beds in high-arrival wards", results_scenario3)
