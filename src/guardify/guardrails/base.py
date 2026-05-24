from abc import ABC, abstractmethod


class GuardrailBase(ABC):

    @abstractmethod
    def predict(self, text: str) -> dict:
        """
        Returns:
            label : "safe" | "injection"
            score : float confidence in [0, 1]
        """
        pass
