from abc import ABCMeta, abstractmethod
from typing import List


class IExecutor(metaclass=ABCMeta):

    @abstractmethod
    def requests(self) -> List[object]:
        """
            Requested types
        return: list of objects.
        """
        pass

    @abstractmethod
    def satisfy(self, reply) -> List[object]:
        """
            Satisfy requested 01.
        :return: list of Req(Enum) contains requests which cannot be satisfied.
        """
        pass

    @abstractmethod
    def start(self, com):
        """
            Do the job.
        """
        pass

    @abstractmethod
    def ready(self) -> bool:
        """
            Is the executor ready for the job.
        """
        pass

    @abstractmethod
    def done(self) -> bool:
        """
            Is job done?
        """
        pass

    @abstractmethod
    def trace_files(self) -> List[str]:
        """
            Return the filename list or executing trace.
        """
        pass
