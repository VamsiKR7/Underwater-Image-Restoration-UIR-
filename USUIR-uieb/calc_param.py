import torch
from net.net import net  # Make sure this import works

# Instantiate model
model = net()

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Trainable Parameters: {total_params:,}')