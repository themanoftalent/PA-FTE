import numpy as np

def compute_edg(risks, event_times, alert_threshold=0.6):
    """EDG = (T_event - T_alert) / T_event if alert before event, else 0"""
    edgs = []
    for r_seq, t_event in zip(risks, event_times):  # per stay
        alert_idx = next((i for i, r in enumerate(r_seq) if r > alert_threshold), None)
        if alert_idx is None or t_event is None:
            edgs.append(0.0)
            continue
        t_alert = alert_idx * 5 / 60  # assume 5-min steps → hours
        lead = t_event - t_alert
        edgs.append(max(0, lead / t_event))
    return np.mean(edgs) if edgs else 0.0


def intervention_utility(leads, fp_count, alpha=1.0, beta=0.5):
    if len(leads) == 0:
        return 0.0
    mean_lead = np.mean(leads)
    far = fp_count / (len(leads) + fp_count)
    return alpha * mean_lead - beta * far
