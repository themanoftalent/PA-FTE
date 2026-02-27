import yaml
import torch

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PA-FTE starting...")

    # 1. Load / preprocess MIMIC-IV data → clients datasets
    # clients = load_mimic_clients(cfg)

    # 2. Init federated encoder
    encoder = CausalTransformerEncoder().to(device)
    risk_head = MemoryAugmentedRisk().to(device)

    # 3. Run federated training (Flower example)
    # strategy = fl.server.strategy.FedProx(...)  # or custom
    # fl.server.start_server(...)

    # 4. Evaluation loop (per patient / stay)
    # for stay in test_stays:
    #     M = torch.zeros(1, cfg["model"]["d_mem"]).to(device)
    #     risks = []
    #     for window in stay_windows:
    #         h = encoder(window)
    #         r, M = risk_head(h, M)
    #         risks.append(r.item())
    #     edg = compute_edg([risks], [stay_event_time])
    #     print(f"EDG: {edg:.3f}")

    # 5. Agent demo
    # agent = PA_FTE_Agent(cfg["agent"]["llm_path"])
    # plan = agent.loop(risks[-1], M.norm().item())
    # print(plan)
