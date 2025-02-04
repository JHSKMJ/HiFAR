import torch
import torch.optim as optim
import torch.nn as nn
from HiFAR_penalty_term import HiFAR_penalty_term


def training(model, device, criterion, optimizer, lamda, epochs, train_loader):
	model = model.to(device)

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		running_acc = 0.0
		for ind, (images, labels) in enumerate(train_loader):
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = model(images)
			_, preds = torch.max(logits, 1)
			task_loss = criterion(logits, labels)
			penalty = HiFAR_penalty_term(model)
			total_loss = task_loss + lamda*penalty
			total_loss.backward()
			optimizer.step()
			running_loss += total_loss.item()*images.size(0)
			running_acc += torch.sum(preds == labels.data)
		epoch_loss = running_loss / len(train_loader.dataset)
		epoch_acc = running_acc / len(train_loader.dataset)
		print(f"epoch-{epoch} || Train Loss : {epoch_loss:.3f}, Train Accuracy : {epoch_acc:.3f}")
	return model

def testing(model, ts_loader, device): 
    model.eval()
    running_acc = 0.

    for ind, (images, labels) in enumerate(ts_loader):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        _, preds = torch.max(logits, 1)
        running_acc += torch.sum(preds == labels.data)
    epoch_acc = running_acc / len(ts_loader.dataset)
    return epoch_acc


