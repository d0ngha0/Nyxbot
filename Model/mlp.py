import torch
import torch.nn as nn

def load_named_models(model_class, model_tags, base_path="./", device='cpu'):
    """
    Load multiple named models (e.g., MLP_RF_Z.pth) into a dictionary.

    Parameters:
    - model_class: class definition of the model (e.g., MLP)
    - model_tags: list of tags like ["RF_Z", "RF_Y", "LF_Z", "LF_Y"]
    - base_path: folder where model files are located
    - device: 'cpu' or 'cuda'

    Returns:
    - Dictionary of loaded models, e.g., models["RF_Z"] -> model
    """
    models = {}
    for tag in model_tags:
        model = model_class()
        path = f"{base_path}MLP_{tag}.pth"
        state_dict = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[tag] = model
    return models
def predict(input_array, model, scaler_X=None, scaler_Y=None):
    model.eval()
    x = input_array.reshape(1, -1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        y_tensor = model(x_tensor)
    y = y_tensor.cpu().numpy().squeeze()
   
    return y
class MLP(nn.Module):
    def __init__(self, input_size=70, hidden_size=32, output_size=70):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)