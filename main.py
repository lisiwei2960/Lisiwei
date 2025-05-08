import torch
from torch import nn, optim
from utils.data_loader import get_data_loaders
from models.TimeXer import TimeXer
from models.TimeXer_SE import TimeXer_SE
from models.TimeXer_Hybrid import TimeXer_Hybrid
import argparse
from tqdm import tqdm
import os


def select_model(model_name, device):
    if model_name == "TimeXer":
        model = TimeXer()
    elif model_name == "TimeXer_SE":
        model = TimeXer_SE()
    elif model_name == "TimeXer_Hybrid":
        model = TimeXer_Hybrid()
    else:
        raise ValueError("Unknown model name.")
    return model.to(device)


def train(model, train_loader, val_loader, device, epochs, save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        val_loss = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved at {save_path}")


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def test(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            predictions.append(output.cpu())
            targets.append(batch_y.cpu())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(targets.numpy(), predictions.numpy())
    rmse = mean_squared_error(targets.numpy(), predictions.numpy(), squared=False)
    print(f"🎯 Test MAE: {mae:.4f} | Test RMSE: {rmse:.4f}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = select_model(args.model, device)
    save_path = f"checkpoints/{args.model}_best.pth"

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    if args.mode == "train":
        os.makedirs("checkpoints", exist_ok=True)
        train(model, train_loader, val_loader, device, epochs=10, save_path=save_path)

    if args.mode == "test":
        model.load_state_dict(torch.load(save_path))
        test(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="train or test")
    parser.add_argument("--model", type=str, choices=["TimeXer", "TimeXer_SE", "TimeXer_Hybrid"], default="TimeXer",
                        help="which model to use")
    args = parser.parse_args()

    main(args)
