import numpy as np

mag_arr = np.load("UV_mags_bhar_sorted_6.5.npy")

# Create a mask for values between -22 and -20
mask = (mag_arr > -22) & (mag_arr < -20)

# Get the indices where the mask is True
indices = np.where(mask)[0]

# Define the number of bins
num_bins = 100  # Example: Create 10 bins between -22 and -20
bins = np.linspace(-22, -20, num_bins + 1)

# Digitize the indices based on the array values
bin_indices = np.digitize(mag_arr[indices], bins)

# Select 10 indices from each bin
selected_indices = []

for bin_num in range(1, num_bins + 1):
    bin_mask = bin_indices == bin_num
    bin_indices_selected = indices[bin_mask]

    # Throw an error if there are fewer than 10 indices in the bin
    if len(bin_indices_selected) < 10:
        raise ValueError(f"Bin {bin_num} has fewer than 10 indices.")

    # Select 10 indices from this bin
    bin_indices_selected = np.random.choice(bin_indices_selected, 10, replace=False)

    selected_indices.extend(bin_indices_selected)

print(len(selected_indices))
np.save("selected_indices_sample_20_22.npy", selected_indices)