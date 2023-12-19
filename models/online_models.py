import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset.dataset import Dataset

from models.base_model import BaseOnlineModel
from models.models_mapping import ModelsMapping


MOST_POPULAR: List[int] = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]


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
        dummy: List[int] = MOST_POPULAR
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


class DatasetEmpoweredRecommendationModel(BaseOnlineModel):
    def __init__(self, model_path: str,
                 dataset_base_path: str = "data/") -> None:
        self.dataset = self.prepare_dataset(dataset_base_path)

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    @staticmethod
    def prepare_dataset(dataset_base_path: str) -> Dataset:
        path = Path(dataset_base_path)

        Columns.Datetime = 'last_watch_dt'
        interactions = pd.read_csv(path / 'interactions.csv')

        interactions[Columns.Datetime] = pd.to_datetime(
            interactions[Columns.Datetime], format='%Y-%m-%d')
        interactions[Columns.Weight] = np.where(
            interactions['watched_pct'] > 10, 3, 1)
        dataset = Dataset.construct(
            interactions_df=interactions
        )
        return dataset

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        reco = self.model.recommend(
            users=[user_id],
            dataset=self.dataset,
            k=k_recs,
            filter_viewed=True,
        ).item_id.tolist()
        return reco


class VectorBasedAnnModel(BaseOnlineModel):
    def __init__(self, model_path: str) -> None:
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_reco(self, user_id: int, k_recs: int) -> List[int]:
        if user_id in self.model.user_id_map.external_ids:
            reco = self.model.get_item_list_for_user(user_id, top_n=k_recs).tolist()
            return reco
        else:
            return MOST_POPULAR


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
        "knn_model": UserKnnModel(models_mapping.userknn_model_path),
        "als_model": DatasetEmpoweredRecommendationModel(models_mapping.als_model_path),
        "lfm_log_model": DatasetEmpoweredRecommendationModel(models_mapping.lfm_logistic_4_model_path),
        "lfm_warp_model": DatasetEmpoweredRecommendationModel(models_mapping.lfm_warp_4_model_path),
        "lfm_log_ann_model": VectorBasedAnnModel(models_mapping.lfm_logistic_4_ann_model_path),
        "lfm_warp_ann_model": VectorBasedAnnModel(models_mapping.lfm_warp_4_ann_model_path),
        "multivae_model": JsonModel(models_mapping.multivae_model_path),
        'ae_attention_model': JsonModel(models_mapping.ae_attention_path)
    }
)
