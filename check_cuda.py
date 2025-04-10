import torch

# Check if CUDA is available
print(f"Is CUDA available: {torch.cuda.is_available()}")

# Get number of CUDA devices
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    # Print info for each device
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("No CUDA devices available")