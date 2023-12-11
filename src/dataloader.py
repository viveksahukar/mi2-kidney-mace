from imports import *
from config import *
class TabularDataset(Dataset):
    def __init__(self, tabular_data, tabular_labels):
        self.tabular_data = torch.FloatTensor(tabular_data.values)
        self.targets = torch.FloatTensor(tabular_labels)  # Rename to targets

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, index):
        return self.tabular_data[index], self.targets[index]

def load_tabular_data(file_path):
    df = pd.read_csv(file_path)
    x = df.drop(columns=['MRN_DEID', 'CardiacFuture_confirmed'])
    y = df['CardiacFuture_confirmed'].values
    return x, y

def load_ecg_data(file_path):
    df = pd.read_csv(file_path)
    # Perform any preprocessing if needed
    return df

def create_data_loader(x, y, batch_size=BATCH_SIZE, shuffle=True):
    dataset = TabularDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def calculate_class_weights(file_path):
    df = pd.read_csv(file_path)
    count_negatives = df['CardiacFuture_confirmed'].value_counts()[0]
    count_positives = df['CardiacFuture_confirmed'].value_counts()[1]
    total_samples = count_negatives + count_positives
    return [total_samples / count_negatives, total_samples / count_positives]

# Load and preprocess data
x_train_tab, y_train_tab = load_tabular_data('../processed-datasets/df_train_tab.csv')
x_val_tab, y_val_tab = load_tabular_data('../processed-datasets/df_val_tab.csv')
x_test_tab, y_test_tab = load_tabular_data('../processed-datasets/df_test_tab.csv')

x_train_ecg = load_ecg_data('../processed-datasets/x_train_ecg.csv')
x_val_ecg = load_ecg_data('../processed-datasets/x_val_ecg.csv')
x_test_ecg = load_ecg_data('../processed-datasets/x_test_ecg.csv')

# Create data loaders
train_loader = create_data_loader(x_train_tab, y_train_tab)
val_loader = create_data_loader(x_val_tab, y_val_tab)
test_loader = create_data_loader(x_test_tab, y_test_tab)

# Calculate class weights
class_weights = calculate_class_weights('../processed-datasets/x_train_ecg.csv')[1]
pos_weight = torch.tensor([class_weights]).to(device)
pos_weight_mod = pos_weight / n

# Calculating category field dimensions
cat_field_dims = [x_train_tab[column].nunique() for column in x_train_tab.columns]
continuous_data = torch.zeros((BATCH_SIZE, 0), device=device)






