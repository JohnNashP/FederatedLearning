import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
from client import FlowerClient
import flwr as fl
from data_utils import create_non_iid_partitions
import sys

print("[Client Init] Preparing data and launching client instance...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
client_datasets = create_non_iid_partitions(dataset, num_clients=10, classes_per_client=2)

client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
train_loader = DataLoader(client_datasets[client_id], batch_size=32, shuffle=True)
test_loader = DataLoader(client_datasets[client_id], batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

print(f"[Client {client_id}] Data ready. Connecting to server at localhost:8080...")

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FlowerClient(model, train_loader, test_loader, device, private_bn=True),
)
