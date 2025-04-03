from abc import ABC, abstractmethod

class BaseWriter(ABC):
    """
    Abstract base class for writers.
    """

    @abstractmethod
    def write(self, key_values, key_excluded, step=0):
        """
        Write key-value pairs to the writer.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the writer.
        """
        pass