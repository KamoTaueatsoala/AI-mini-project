import h5py

# Path to your HDF5 file
h5_file_path = "all_mias_scans.h5"

# Open the file in read mode
with h5py.File(h5_file_path, "r") as f:
    # Function to recursively print keys and dataset shapes
    def print_h5_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}")
        else:
            print(f"Group: {name}")
    
    f.visititems(print_h5_structure)
