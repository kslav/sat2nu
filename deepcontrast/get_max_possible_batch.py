import torch

# Assuming 'model' is your PyTorch model and 'device' is your GPU
model.to(device)
model.eval()

with torch.no_grad():  # Ensures that gradients are not computed
    # Start with a small batch size and increase it until you find the maximum size that fits
    batch_size = 32  # Example starting point, adjust based on your initial experiments
    while True:
        try:
            # Assuming 'get_batch' is a function that returns a batch of input data of size 'batch_size'
            inputs = get_batch(batch_size).to(device)
            outputs = model(inputs)
            print(f"Batch size {batch_size} fits in the GPU")
            batch_size *= 2  # Double the batch size for the next iteration
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} is too large")
                break
            else:
                raise e
