from abc import ABC
from typing import List


class BaseOnlineModel(ABC):
    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        pass
