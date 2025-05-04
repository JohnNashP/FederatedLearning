import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from utils import train, test, get_model_parameters, set_model_parameters

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, private_bn):
        print("[Client] Initializing FlowerClient instance...")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.private_bn = private_bn
        print("[Client] Initialization complete.")

    def get_parameters(self, config):
        print("[DEBUG] get_parameters() called by server")
        return [val.cpu().numpy() for val in get_model_parameters(self.model, include_bn=True).values()]

    def set_parameters(self, parameters):
        print("[Client] Receiving model parameters from server...")
        params = list(get_model_parameters(self.model, include_bn=True).keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(params, parameters)}
        set_model_parameters(self.model, state_dict)

    def fit(self, parameters, config):
        print("\n[Client] Starting local training phase...")
        self.set_parameters(parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        train(self.model, self.train_loader, epochs=1, optimizer=optimizer, criterion=criterion, device=self.device)
        print("[Client] Local training complete. Sending updated weights to server.\n")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("\n[Client] Starting evaluation phase...")
        self.set_parameters(parameters)
        acc = test(self.model, self.test_loader, device=self.device)
        print("[Client] Evaluation complete.\n")
        return 0.0, len(self.test_loader.dataset), {"accuracy": acc}
