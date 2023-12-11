from imports import *
from dataloader import *
from config import *
from models import *


# TensorBoard Setup
PATH = './resources/'
TAG = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
writer = SummaryWriter(os.path.join(PATH, 'logs/%s_log' % TAG))

# Initialize model and optimizer
model = create_model()
optimizer = configure_optimizer(model)

# Loss Function
loss = torch.nn.BCELoss(weight=pos_weight_mod)

# Training Loop
train_losses, test_losses = [], []
for epoch in range(n_epochs):
    train_loss = train_tool(model, optimizer, train_loader, loss, device=device)
    test_loss, _, _ = test_tool(model, val_loader, loss, device=device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'\n Epoch {epoch + 1}  '
          f'train loss:{round(train_loss, 4)}. '
          f'test loss:{round(test_loss, 4)}')
    writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, global_step=epoch)

writer.close()

# Save Model Weights
torch.save(model.state_dict(), '../results/saved_models/model_final.pth')