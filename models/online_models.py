import json
import pickle

from models.base_model import BaseOnlineModel
from typing import List

from models.models_mapping import ModelsMapping


class DummyModel(BaseOnlineModel):
    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        dummy: List[int] = list(range(k_recs))
        return dummy


class JsonModel(BaseOnlineModel):
    def __init__(self, model_path: str) -> None:
        hook = lambda x: {int(k): v for k, v in x.items()}
        with open(model_path, "r", encoding="utf8") as file:
            self.model = json.load(file, object_hook=hook)

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        dummy: List[int] = list(range(k_recs))
        reco = self.model.get(user_id)

        if reco is None:
            default = self.model.get("default")
            return default if default is not None else dummy

        return reco


class UserKnnModel(BaseOnlineModel):
    def __init__(self, model_path: str) -> None:
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        reco = self.model.recommend(user_id=user_id, k_recs=k_recs)
        return reco


class AvailableModelsDict(dict):
    def get(self, key, default=None, exception=None):
        res = super().get(key, default)
        if res is None:
            raise exception
        return res


models_mapping = ModelsMapping()
AVAILABLE_MODELS = AvailableModelsDict(
    {
        "simple_model": DummyModel(),
        "popular_model": JsonModel(models_mapping.popular_model_path),
        "knn_model": UserKnnModel(models_mapping.userknn_model_path)
    }
)
