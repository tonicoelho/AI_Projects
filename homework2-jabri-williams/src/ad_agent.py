"""
ad_engine.py
Advertisement Selection Engine that employs a Decision Network (Bayesian Network)
to Maximize Expected Utility (MEU), compute Value of Perfect Information (VPI),
and infer most likely consumer profiles based on partial evidence.
"""

import itertools
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.inference.CausalInference import CausalInference

class AdEngine:
    def __init__(self, data: pd.DataFrame, structure: list[tuple[str, str]],
                 dec_vars: list[str], util_map: dict[str, dict[int, int]]):
        valid_cols = set(data.columns)
        self.structure = [(p, c) for (p, c) in structure if p in valid_cols and c in valid_cols]
        self.model = BayesianNetwork(self.structure)
        self.domains = {col: sorted(data[col].unique()) for col in data.columns}
        self.model.fit(data, estimator=MaximumLikelihoodEstimator, state_names=self.domains)

        self.ve_infer = VariableElimination(self.model)
        self.causal_infer = CausalInference(self.model)

        self.dec_vars = dec_vars
        self.util_map = util_map
        self.all_vars = set(data.columns)
        self.chance_vars = self.all_vars - set(self.dec_vars)

    def meu(self, evidence: dict[str, int]) -> tuple[dict[str, int], float]:
        best_utility = Decimal('-inf')
        best_assignment = {}

        dec_combinations = itertools.product(*[self.domains[d] for d in self.dec_vars])

        for combo in dec_combinations:
            dec_assignment = dict(zip(self.dec_vars, combo))
            total_utility = Decimal('0')

            for util_var, payoff in self.util_map.items():
                qres = self.causal_infer.query(
                    variables=[util_var],
                    do=dec_assignment,
                    evidence=evidence,
                    inference_algo='ve',
                    show_progress=False
                )
                probs = qres.values
                states = self.domains[util_var]
                for i, state in enumerate(states):
                    prob = Decimal(str(probs[i]))
                    utility = Decimal(str(payoff.get(state, 0)))
                    total_utility += prob * utility
                    # Logging for debugging
                    # print(f"Evidence={evidence}, Combo={dec_assignment}, S={state}, P={probs[i]}, U={utility}")

            if total_utility > best_utility:
                best_utility = total_utility
                best_assignment = dec_assignment

        # Convert to float with exact 2-decimal precision
        return best_assignment, float(best_utility.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def vpi(self, potential_evidence: str, observed_evidence: dict[str, int]) -> float:
        _, baseline_meu = self.meu(observed_evidence)
        baseline_meu = Decimal(str(baseline_meu))

        qres = self.ve_infer.query(
            variables=[potential_evidence],
            evidence=observed_evidence,
            show_progress=False
        )
        probs = qres.values
        states = self.domains[potential_evidence]

        weighted_meu = Decimal('0')
        for i, state in enumerate(states):
            new_evidence = observed_evidence.copy()
            new_evidence[potential_evidence] = state
            _, new_meu = self.meu(new_evidence)
            weighted_meu += Decimal(str(probs[i])) * Decimal(str(new_meu))

        vpi_value = max(Decimal('0'), weighted_meu - baseline_meu)
        return float(vpi_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def most_likely_consumer(self, evidence: dict[str, int]) -> dict[str, int]:
        missing_vars = [var for var in self.chance_vars if var not in evidence]

        if not missing_vars:
            return evidence.copy()

        map_res = self.ve_infer.map_query(
            variables=missing_vars,
            evidence=evidence,
            show_progress=False
        )

        result = evidence.copy()
        result.update({k: int(v) for k, v in map_res.items()})
        return result