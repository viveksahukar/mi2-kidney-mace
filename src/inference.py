from imports import *
from models import *
from dataloader import *
from plotting import *
from config import *


def load_model(model_path, cat_field_dims):
    # Initialize model
    model = create_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model('../results/saved_models/model_final.pth', cat_field_dims)
model.to(device)
loss = torch.nn.BCELoss(weight=pos_weight_mod)

# Evaluate model performance
_, y_true, y_pred_probs = test_tool(model, val_loader, loss, device=device)

plot_training_validation_loss(train_losses, test_losses)
plot_roc_curve(y_true, y_pred_probs)
plot_precision_recall_curve(y_true, y_pred_probs)

# Save model weights
torch.save(model.state_dict(), '../results/saved_models/model_final.pth')