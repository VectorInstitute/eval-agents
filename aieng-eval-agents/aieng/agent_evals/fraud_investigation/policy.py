"""Defines triage policies for fraud investigation alerts."""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


class TriageOutcome(str, Enum):
    """Possible triage outcomes for fraud investigation alerts."""

    CLOSE = "CLOSE"
    INVESTIGATE = "INVESTIGATE"
    ESCALATE = "ESCALATE"


class TriageStrategy(ABC):
    """Abstract base class for triage strategies."""

    @abstractmethod
    async def triage(self, alert: dict) -> TriageOutcome:
        """Return the triage outcome for the given alert.

        Parameters
        ----------
        alert : dict
            The alert data to be triaged.

        Returns
        -------
        TriageOutcome
            The outcome of the triage process.
        """
        pass


class TriagePolicyConfig(BaseModel):
    """Configuration for rule-based triage policy."""

    auto_close_threshold: int = 35
    auto_escalate_threshold: int = 95


class RuleBasedTriage(TriageStrategy):
    """A simple rule-based triage strategy based on risk score.

    This simply compares the alert's risk score against predefined thresholds
    to determine the triage outcome.

    Parameters
    ----------
    config : TriagePolicyConfig
        Configuration parameters for the triage policy.
    """

    def __init__(self, config: TriagePolicyConfig) -> None:
        self.config = config

    async def triage(self, alert: dict) -> TriageOutcome:
        """Triage the alert based on its risk score.

        Parameters
        ----------
        alert : dict
            The alert data to be triaged. Should contain a 'risk_score' key.

        Returns
        -------
        TriageOutcome
            The outcome of the triage process.
        """
        score = alert.get("risk_score", 0)
        if score < self.config.auto_close_threshold:
            return TriageOutcome.CLOSE
        if score > self.config.auto_escalate_threshold:
            return TriageOutcome.ESCALATE
        return TriageOutcome.INVESTIGATE
