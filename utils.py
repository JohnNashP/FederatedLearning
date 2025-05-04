import torch

def get_model_parameters(model, include_bn=True):
    state_dict = model.state_dict()
    if include_bn:
        return {k: v.clone() for k, v in state_dict.items()}
    return {k: v.clone() for k, v in state_dict.items() if 'bn' not in k}

def set_model_parameters(model, params):
    model.load_state_dict(params, strict=False)

def train(model, train_loader, epochs, optimizer, criterion, device):
    model.train()
    for epoch in range(epochs):
        print(f"\n[Training Epoch {epoch+1}] Starting local training...")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print(f"  - Batch {batch_idx}: Loss = {loss.item():.4f}")
        print(f"[Training Epoch {epoch+1}] Completed.\n")

def test(model, test_loader, device):
    print("[Evaluation] Running local model evaluation...")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"[Evaluation] Accuracy: {acc:.4f}")
    return acc