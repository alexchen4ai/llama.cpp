import torch
import torch.nn as nn
import os
from torchinfo import summary

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MLP Layer
class MLP(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, output_size=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float32)  
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.relu = nn.ReLU() 
        torch.manual_seed(43)  

        # Initialize weights
        self.fc1.weight.data = torch.randn(self.fc1.weight.size()) * 0.1
        self.fc2.weight.data = torch.randn(self.fc2.weight.size()) * 0.1
        self.fc1.bias.data = torch.randn(self.fc1.bias.size()) * 0.1
        self.fc2.bias.data = torch.randn(self.fc2.bias.size()) * 0.1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def print_model_architecture_and_weights(model):
    print("Model Architecture:")
    print(model)
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        print(f"{name} device: {param.device}")
        if 'weight' in name:
            print(f"\n{name} shape: {param.shape}")
            print(param.data)
        elif 'bias' in name:
            print(f"\n{name} shape: {param.shape}")
            print(param.data)


def inference(model):
    sample_input = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32).to(device)
    sample_input = sample_input.view(1, -1)
    print(f"Sample input:\n{sample_input}")
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
        print(f"Model output:\n{output}")

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    model = MLP().to(device)

    # Load or save the model
    model_path = f'model/two_layer_mlp.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

    print("\nModel Summary:")
    summary(
        model,
        input_size=(1, 5),  # (batch_size, input_size)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        device=device,  
    )

    import time
    start = time.time()
    inference(model)
    print(f"Inference time: {time.time() - start:.4f} seconds")