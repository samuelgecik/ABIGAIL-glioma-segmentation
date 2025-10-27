from brats_mvp.model import UNet
from dataManager import get_training_data

import torch
import torch.nn as nn
from tqdm import tqdm

def main():
    model = UNet(in_channels=1, out_classes=2)
    training_loader, validation_loader = get_training_data(val_split=0.2)
    loss_function = nn.CrossEntropyLoss() #DICE
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9)

    #training loop
    epochs = 2
    device = torch.device("cpu")

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0

        # Training loop with progress bar
        train_pbar = tqdm(training_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
        for images, labels in train_pbar:
            images = images.to(device)  # [B, C, H, W]
            labels = labels.squeeze(1).long().to(device)  # [B, H, W]

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)  # [B, K, H, W]
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_ce = training_loss / max(1, len(training_loader))

        model.eval()
        val_ce = 0.0
        
        # Validation loop with progress bar
        val_pbar = tqdm(validation_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", unit="batch")
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.squeeze(1).long().to(device)
                predictions = model(images)
                loss = loss_function(predictions, labels)
                val_ce += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        val_ce /= max(1, len(validation_loader))
        
        print(f"[{epoch:02d}] train_ce={train_ce:.4f} | val_ce={val_ce:.4f}")



if __name__ == "__main__":
    main()