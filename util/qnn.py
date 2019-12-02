import torch
import torch.nn as nn

class Q(nn.Module):
    """Q-network using a NN"""
    def __init__(self, state_dim, action_dim, lr):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fitted = False
        
        self.model = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def forward(self, state):
        """Forward"""
        state = torch.tensor(state).cuda().float()
        return self.model(state)
    
    def predict(self, state):
        """Forward without gradients (used for predictions)"""
        state = torch.tensor(state).cuda().float()
        with torch.no_grad():
            return self.model(state).squeeze().cpu().numpy()
    
    def fit(self, state, true_value):
        """Fit NN with a single backward step"""
        self.fitted = True
        state = torch.tensor(state).cuda().float()
        true_value = torch.tensor(true_value).cuda().float()
        self.optimizer.zero_grad()
        out = self(state).squeeze()
        loss = self.criterion(out, true_value)
        loss.backward()
        self.optimizer.step()