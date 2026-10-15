# federated/fedprox_training.py

import flwr as fl
from flwr.common import Metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from models.encoder import CausalTransformerEncoder   # from earlier

class FedProxClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        net: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        mu: float = 0.01,
        local_epochs: int = 2,
        lr: float = 1e-4
    ):
        self.cid = cid
        self.net = net.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.mu = mu
        self.local_epochs = local_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.from_numpy(np.array(v)) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        global_params = [p.detach().clone() for p in self.net.parameters()]

        self.net.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                h = self.net(x)               # [B, d_model]
                logit = h.mean(dim=-1)        # simple head; replace with paper's logistic
                loss = self.criterion(logit.squeeze(), y)

                # FedProx proximal term
                prox = 0.0
                for p, gp in zip(self.net.parameters(), global_params):
                    prox += (self.mu / 2.0) * torch.sum((p - gp.to(self.device)) ** 2)
                total_loss = loss + prox
                total_loss.backward()
                optimizer.step()

        return len(self.trainloader.dataset), self.get_parameters({}), {"loss": total_loss.item() / len(self.trainloader)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.valloader:
                x, y = x.to(self.device), y.to(self.device)
                h = self.net(x)
                logit = h.mean(dim=-1)
                loss += self.criterion(logit.squeeze(), y).item()
                pred = (torch.sigmoid(logit) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        return loss / len(self.valloader), total, {"accuracy": accuracy}


def start_fedprox(
    num_clients: int = 5,
    num_rounds: int = 50,
    fraction_fit: float = 0.6,
    mu: float = 0.01
):
    # Dummy client creation - replace with real partitioned datasets
    def client_fn(cid: str) -> fl.client.Client:
        net = CausalTransformerEncoder()
        # Replace with real loaders
        trainloader = DataLoader([], batch_size=32)  # ← TODO: your IrregularTimeSeriesDataset
        valloader   = DataLoader([], batch_size=32)
        return FedProxClient(cid, net, trainloader, valloader, torch.device("cuda"), mu=mu)

    strategy = fl.server.strategy.FedAvg(  # FedProx is client-side → use FedAvg wrapper
        fraction_fit=fraction_fit,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=lambda metrics: {"accuracy": sum(m["accuracy"] * n for m, n in metrics) / sum(n for _, n in metrics)}
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2}  # adjust
    )
