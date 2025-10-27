from brats_mvp.model import UNet
from dataManager import get_training_data, get_validation_data

import torch
import torch.nn as nn

def main():
    model = UNet(in_channels=1, out_classes=2)
    training_loader = get_training_data()
    validation_loader = get_validation_data()
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

        for images, labels in training_loader:
            images = images.to(device)  # [B, C, H, W]
            labels = labels.squeeze(1).long().to(device)  # [B, H, W]

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)  # [B, K, H, W]
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        train_ce = training_loss / max(1, len(training_loader))

        model.eval()
        val_ce = 0.0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.squeeze(1).long().to(device)
                predictions = model(images)
                val_ce += loss_function(predictions, labels).item()
        val_ce /= max(1, len(validation_loader))
        print(f"[{epoch:02d}] train_ce={train_ce:.4f} | val_ce={val_ce:.4f}")



if __name__ == "__main__":
    main()