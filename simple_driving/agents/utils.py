import torch

def save_model(model, filename="dqn_weights.pth"):
    torch.save(model.state_dict(), filename)

def load_model(model, filename="dqn_weights.pth"):
    model.load_state_dict(torch.load(filename))
    model.eval()
