# Requires: pip install flwr
import flwr as fl
import torch
from torch.utils.data import DataLoader
# assume Dataset class defined elsewhere

class PA_FTE_Client(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, device):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=config.get("lr", 1e-4))
        mu = config.get("mu", 0.01)
        global_params = [p.clone().detach() for p in self.net.parameters()]

        self.net.train()
        for epoch in range(config.get("local_epochs", 2)):
            for batch in self.trainloader:
                x, delta, m, y = [b.to(self.device) for b in batch]
                # preprocess → x already prepared [B, L, d]
                h = self.net(x, mask=m)
                logit = ...  # add head if needed; paper uses logistic on h_t + M
                loss = self.criterion(logit.squeeze(), y.float())
                prox = 0.0
                for p, gp in zip(self.net.parameters(), global_params):
                    prox += (mu / 2) * torch.sum((p - gp.to(self.device))**2)
                (loss + prox).backward()
                optimizer.step()
                optimizer.zero_grad()
        return len(self.trainloader.dataset), self.get_parameters({}), {}

    def evaluate(self, parameters, config):
        # implement validation AUROC, etc.
        return 0.0, len(self.valloader.dataset), {"auroc": 0.0}
