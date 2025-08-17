'''
Skeleton for answering queries related to the Ad Agent.
'''

from ad_agent import AdEngine
from constants import *
import numpy as np
import pandas as pd

class AdAgentQueries:
    """
    See Problem 7 in the Spec for requested answer formats below
    """

    def __init__(self, ad_agent: "AdEngine") -> None:
        self._ad_agent = ad_agent

    def answer_7_1(self) -> float:
        """
        For Problem 7.1: Return the VPI of gun control stance (G).
        Suppose we have no prior evidence about the consumer.
        """
        return self._ad_agent.vpi("G", {})

    def answer_7_2(self) -> float:
        """
        For Problem 7.2: Return the VPI of 'P' given we know 'G'=1
        from the problem statement
        """
        evidence = {"G": 1}
        return self._ad_agent.vpi("P", evidence)

    def answer_7_3(self) -> dict[str, int]:
        """
        For Problem 7.3: Return the most likely consumer characteristics
        if we know they are a millennial and a home owner
        (A=0 => millennial, H=1 => owner)
        """
        # Example evidence => "A":0, "H":1
        evidence = {"A": 0, "H": 1}
        profile = self._ad_agent.most_likely_consumer(evidence)
        return profile


if __name__ == '__main__':
    """
    Use this main method to run the requested queries for your report.
    """
    ad_engine = AdEngine(ADBOT_DATA, ADBOT_STRUC, ADBOT_DEC, ADBOT_UTIL)
    querier = AdAgentQueries(ad_engine)

    print("Answer to 7.1 (VPI(G) with no evidence):", querier.answer_7_1())
    print("Answer to 7.2 (VPI(P) given G=1):", querier.answer_7_2())
    print("Answer to 7.3 (most likely consumer if A=0, H=1):", querier.answer_7_3())
