from llama_cpp import Llama
import json

class PA_FTE_Agent:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False)

    def generate_plan(self, state_dict):
        state_json = json.dumps(state_dict)
        prompt = f"""Current state: {state_json}

Generate JSON plan: {{"risk_level": str, "recommended_action": str, "rationale": str, ...}}
Use local tools if needed."""

        output = self.llm(prompt, max_tokens=400, temperature=0.6, stop=["}"])
        try:
            plan_text = output["choices"][0]["text"].strip() + "}"
            plan = json.loads(plan_text)
            return plan
        except:
            return {"risk_level": "unknown", "recommended_action": "alert clinician"}

    def loop(self, r_aug, M_norm, ...):
        if r_aug <= 0.45:
            return None
        state = {"risk": float(r_aug), "belief": float(M_norm), ...}
        candidates = [self.generate_plan(state) for _ in range(5)]
        # score by utility, validate, return best
        return max(candidates, key=lambda p: p.get("confidence", 0))
