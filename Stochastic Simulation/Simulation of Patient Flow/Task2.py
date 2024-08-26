import numpy as np
import pandas as pd
from scipy.stats import erlang

import random
np.random.seed(42)

# Parameters
wards = ['A', 'B', 'C', 'D', 'E', 'F']
arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0, 'F': 13.0}
mean_stay = {'A': 2.9, 'B': 4.0, 'C': 4.5, 'D': 1.4, 'E': 3.9, 'F': 2.2}
initial_capacities = {'A': 55, 'B': 40, 'C': 30, 'D': 20, 'E': 20}
urgency_points = {'A': 7, 'B': 5, 'C': 2, 'D': 10, 'E': 5}
relocation_probs = {
    'A': [0.00, 0.05, 0.10, 0.05, 0.80],
    'B': [0.20, 0.00, 0.50, 0.15, 0.15],
    'C': [0.30, 0.20, 0.00, 0.20, 0.30],
    'D': [0.35, 0.30, 0.05, 0.00, 0.30],
    'E': [0.20, 0.10, 0.60, 0.10, 0.00]
}

# Determine the minimum bed capacity for Ward F
arrival_rate_F = arrival_rates['F']
mean_stay_F = mean_stay['F']

def erlang_b(n, traffic_intensity):
    """ Compute the Erlang B formula """
    inv_b = 1.0
    for j in range(1, n + 1):
        inv_b = 1 + inv_b * j / traffic_intensity
    return 1.0 / inv_b

# Calculate traffic intensity for Ward F
traffic_intensity_F = arrival_rate_F * mean_stay_F

# Find the minimum capacity such that the blocking probability is <= 5%
bed_capacity_F = 0
for n in range(1, 100):  # Arbitrary upper limit for bed capacity
    if erlang_b(n, traffic_intensity_F) <= 0.05:
        bed_capacity_F = n
        break

# Allocate beds to Ward F and adjust other wards based on urgency points
def reallocate_beds(initial_capacities, urgency_points, bed_capacity_F):
    # Calculate the total available beds
    total_beds = sum(initial_capacities.values())
    
    # Reduce the total beds by the number of beds allocated to Ward F
    remaining_beds = total_beds - bed_capacity_F
    
    # Sort wards by urgency points (highest priority first)
    sorted_wards = sorted(initial_capacities.keys(), key=lambda x: urgency_points[x], reverse=True)
    
    # Distribute remaining beds to wards based on initial proportions
    adjusted_capacities = {}
    for ward in sorted_wards:
        proportion = initial_capacities[ward] / total_beds
        adjusted_capacities[ward] = int(proportion * remaining_beds)
    
    # Ensure the total beds match the remaining beds
    current_total = sum(adjusted_capacities.values())
    difference = remaining_beds - current_total
    if difference > 0:
        for ward in sorted_wards:
            adjusted_capacities[ward] += 1
            difference -= 1
            if difference == 0:
                break
    
    return adjusted_capacities

# Add relocation probabilities for new Ward F
def update_relocation_probs(relocation_probs):
    for ward in relocation_probs:
        relocation_probs[ward].append(0.0 if ward != 'F' else 1.0)
    relocation_probs['F'] = [0.20, 0.20, 0.20, 0.20, 0.20, 0.0]
    return relocation_probs

# Initialize the state of the system
def initialize_ward_occupancy(adjusted_capacities):
    return {ward: 0 for ward in adjusted_capacities}

# Function to run the simulation
def simulate_hospital_with_new_ward(days, adjusted_capacities, relocation_probs):
    ward_occupancy = initialize_ward_occupancy(adjusted_capacities)
    total_admissions = {ward: 0 for ward in adjusted_capacities}
    total_relocations = {ward: 0 for ward in adjusted_capacities}
    total_losses = {ward: 0 for ward in adjusted_capacities}
    
    for day in range(days):
        for ward in adjusted_capacities:
            arrivals = np.random.poisson(arrival_rates[ward])
            for _ in range(arrivals):
                if ward_occupancy[ward] < adjusted_capacities[ward]:
                    ward_occupancy[ward] += 1
                    total_admissions[ward] += 1
                else:
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

            # Handle patient departures based on length of stay
            departures = np.random.poisson(ward_occupancy[ward] / mean_stay[ward])
            ward_occupancy[ward] = max(0, ward_occupancy[ward] - departures)
    
    return total_admissions, total_relocations, total_losses, ward_occupancy

# Find the optimal bed capacity for Ward F to ensure 95% hospitalization rate
def find_optimal_bed_capacity_for_f(days, target_rate=0.95):
    for bed_capacity in range(1, 101):  # Arbitrary upper limit for bed capacity
        adjusted_capacities = reallocate_beds(initial_capacities, urgency_points, bed_capacity)
        adjusted_capacities['F'] = bed_capacity
        
        relocation_probs_updated = update_relocation_probs(relocation_probs.copy())
        total_admissions, total_relocations, total_losses, final_occupancy = simulate_hospital_with_new_ward(days, adjusted_capacities, relocation_probs_updated)
        hospitalization_rate_F = total_admissions['F'] / (total_admissions['F'] + total_losses['F'])
        
        if hospitalization_rate_F >= target_rate:
            return bed_capacity, hospitalization_rate_F, adjusted_capacities
    
    return None, None, None  # If no suitable capacity is found within the range

# Find the minimum bed capacity for Ward F
optimal_bed_capacity, hospitalization_rate, adjusted_capacities = find_optimal_bed_capacity_for_f(365)
print(f"Optimal bed capacity for Ward F: {optimal_bed_capacity}")
print(f"Hospitalization rate for Ward F: {hospitalization_rate}")

# Run the simulation with the adjusted capacities
relocation_probs_updated = update_relocation_probs(relocation_probs.copy())
total_admissions, total_relocations, total_losses, final_occupancy = simulate_hospital_with_new_ward(365, adjusted_capacities, relocation_probs_updated)

# Display results
results = pd.DataFrame({
    'Admissions': total_admissions,
    'Relocations': total_relocations,
    'Losses': total_losses,
    'Final Occupancy': final_occupancy
})
print(results)
