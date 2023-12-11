from imports import *
from config import *
from dataloader import *
def create_model():
    """Create and initialize the TabTransformer model using settings from config."""
    model = TabTransformer(cat_field_dims, cons_dims=cons_dims,
                           embed_dim=embed_dim, depth=depth,
                           n_heads=n_heads, att_dropout=att_dropout,
                           an_dropout=an_dropout, ffn_dropout=ffn_dropout,
                           mlp_dims=mlp_dims)
    return model.to(device)

def configure_optimizer(model):
    """Configure the optimizer for the model using settings from config."""
    weight_parameters, bias_parameters = [], []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_parameters.append(param)
        if 'bias' in name:
            bias_parameters.append(param)

    return Adam(params=[{'params': weight_parameters, 'weight_decay': weight_decay},
                        {'params': bias_parameters}], lr=learning_rate)

class TabularDataset(Dataset):
    def __init__(self, tabular_data, tabular_labels):
        self.tabular_data = torch.FloatTensor(tabular_data.values)
        self.targets = torch.FloatTensor(tabular_labels)  # Rename to targets

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, index):
        return self.tabular_data[index], self.targets[index]

class ScaleDotProductAttention(torch.nn.Module):
    def __init__(self, dropout=0.5, **kwargs):
        super(ScaleDotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        d = q.size()[-1]
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / (d ** 0.5)
        if mask is not None:
            attn_scores = torch.masked_fill(attn_scores, mask == 0, -1e9)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        attn_output = torch.matmul(attn_scores, v)
        return attn_output

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_head, dropout=0.5, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.n_head = n_head
        self.head_dim = input_dim // n_head
        self.q_w = torch.nn.Linear(input_dim, n_head * self.head_dim, bias=False)
        self.k_w = torch.nn.Linear(input_dim, n_head * self.head_dim, bias=False)
        self.v_w = torch.nn.Linear(input_dim, n_head * self.head_dim, bias=False)
        self.fc = torch.nn.Linear(n_head * self.head_dim, input_dim, bias=False)
        self.attention = ScaleDotProductAttention(dropout=dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, input_dim = q.size()
        q = self.q_w(q).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_w(k).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_w(v).view(batch_size, seq_len, self.n_head, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        attn_out = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.fc(attn_out)
        return out

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(torch.nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class AddNormConnection(torch.nn.Module):
    def __init__(self, dim, dropout):
        super(AddNormConnection, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, layer_out):
        x = x + self.dropout(layer_out)
        return self.layer_norm(x)

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, mult = 4, dropout=0.5, ff_act='GEGLU'):
        super(FeedForward, self).__init__()
        if ff_act == 'GEGLU':
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim * mult * 2),
                GEGLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_dim * mult, input_dim)
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim * mult),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_dim * mult, input_dim)
            )

    def forward(self, x):
        return self.net(x)

class TabTransformerEncoderBlock(torch.nn.Module):
    def __init__(self, input_dim, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout):
        super(TabTransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(input_dim, n_head=n_heads, dropout=att_dropout)
        self.ffn = FeedForward(input_dim, mult=ffn_mult, dropout=ffn_dropout, ff_act=ffn_act)
        self.add_norm1 = AddNormConnection(input_dim, dropout=an_dropout)
        self.add_norm2 = AddNormConnection(input_dim, dropout=an_dropout)

    def forward(self, x):
        '''
        encoder block
        :param x: embed_x
        :return:
        '''
        att_out = self.attention(x, x, x)
        add_norm1_out = self.add_norm1(x, att_out)
        ffn_out = self.ffn(add_norm1_out)
        out = self.add_norm2(add_norm1_out, ffn_out)
        return out

class TabTransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, depth, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout):
        super(TabTransformerEncoder, self).__init__()
        transformer = []
        for _ in range(depth):
            transformer.append(TabTransformerEncoderBlock(input_dim, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout))
        self.transformer = torch.nn.Sequential(*transformer)

    def forward(self, x):
        '''
        encoder block
        :param x: embed_x
        :return:
        '''
        out = self.transformer(x)
        return out

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)
        self.offsets = torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.int64)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x.long()
        x = x + self.offsets.unsqueeze(0).to(x.device)
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, dropout=0.5, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        for layer_dim in layer_dims:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.BatchNorm1d(layer_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TabTransformer(torch.nn.Module):
    def __init__(self, cat_field_dims, cons_dims, embed_dim, depth=2, n_heads=4, att_dropout=0.5, ffn_mult=2, ffn_dropout=0.5, ffn_act='GEGLU', an_dropout=0.5, mlp_dims=[10, 10], mlp_dropout=0.5):
        super(TabTransformer, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims=cat_field_dims, embed_dim=embed_dim)
        self.transformer = TabTransformerEncoder(embed_dim, depth, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout)
        self.embed_output_dim = len(cat_field_dims) * embed_dim #+ max(cons_dims, 1)  # ensure at least size 1
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, layer_dims=mlp_dims, dropout=mlp_dropout)
        self.norm = None
        if cons_dims > 0:
            self.norm = torch.nn.LayerNorm(cons_dims)


    def forward(self, x_cat, x_cons):
        embed_x = self.embedding(x_cat)
        trans_out = self.transformer(embed_x)
        if x_cons is not None and self.norm is not None:
            cons_x = self.norm(x_cons)
            all_x = torch.cat([trans_out.flatten(1), cons_x], dim=-1)
        else:
            all_x = trans_out.flatten(1)
        # cons_x = self.norm(x_cons)
        # all_x = torch.cat([trans_out.flatten(1), cons_x], dim=-1)
        out = self.mlp(all_x)
        return torch.sigmoid(out.squeeze(1)) # do you need this, modify this for concatenation with ecg_out?

def predict_prob(model, data_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch1, X_batch2, y_batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            X_batch1, X_batch2, y_batch = X_batch1.to(device), X_batch2.to(device), y_batch.to(device)
            y_out = model(X_batch1, X_batch2)
            y_pred.extend(y_out.tolist())
    return y_pred

def train_tool(model, optimizer, data_loader, criterion, device, log_interval=0):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (tabular_data, tabular_labels) in enumerate(tk0):
        tabular_data, tabular_labels = tabular_data.to(device), tabular_labels.to(device)
        # Instead of 0, pass a tensor with the appropriate shape
        continuous_data = torch.zeros((tabular_data.size(0), 0), device=device)  # Create an empty tensor for continuous data
        expected_input_dim = sum(cat_field_dims)
        actual_input_dim = tabular_data.shape[1]  # Assuming tabular_data shape is (batch_size, num_features)

        # # Check if actual input dimension matches the expected dimension
        # if actual_input_dim != expected_input_dim:
        #     raise ValueError(f"Input dimension mismatch. Expected dimension: {expected_input_dim}, Actual dimension: {actual_input_dim}")

        y_out = model(tabular_data, continuous_data)

        loss = criterion(y_out, tabular_labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if log_interval:
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
    return total_loss / len(data_loader)

def test_tool(model, data_loader, criterion, device):
    model.eval()
    y_true, y_pred, y_pred_probs = [], [], []
    total_loss = 0
    with torch.no_grad():
        for tabular_data, tabular_labels in data_loader:
            tabular_data, tabular_labels = tabular_data.to(device), tabular_labels.to(device)
            y_out = model(tabular_data, continuous_data)
            loss = criterion(y_out, tabular_labels.float())
            y_true.extend(tabular_labels.tolist())
            y_pred.extend(y_out.tolist())
            y_pred_probs.extend(y_out.sigmoid().tolist())  # Sigmoid to get probabilities
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss, y_true, y_pred_probs
