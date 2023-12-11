from imports import *
from dataloader import *
from config import *
from models import *

# Optuna training
def objective(trial):
    # Define the hyperparameter space
    # batch_size = trial.suggest_categorical('batch_size', [20, 40, 60, 80])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    pos_weight_power = trial.suggest_categorical('pos_weight_power', [4, 6, 8, 10])
    n_epochs = trial.suggest_categorical('n_epochs', [25, 50, 75, 100])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    loss = nn.BCELoss(weight=pos_weight_mod)

    model = create_model()
    # Define the optimizer
    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ... (Include the rest of your model training and validation code here, using the train_tool and test_tool functions)
    for epoch in range(n_epochs):
        train_loss = train_tool(model, optimizer, train_loader, loss, device=device)
        test_loss, _, _ = test_tool(model, val_loader, loss, device=device)

        # For Optuna, we can report intermediate results
        trial.report(test_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the final test loss
    final_test_loss, _, _ = test_tool(model, val_loader, loss, device=device)
    return final_test_loss

# Create a study object and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)

# Print and visualize the results
trial = study.best_trial
print(f"Best trial: Loss: {trial.value}, Params: {trial.params}")

# Show the optimization results (if using Jupyter Notebook or similar environment)
# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_slice(study)
# optuna.visualization.plot_parallel_coordinate(study)

# tensorboard
# PATH = './resources/'
# TAG = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# writer = SummaryWriter(os.path.join(PATH, 'logs/%s_log' % TAG))
# best_thresholds = []

