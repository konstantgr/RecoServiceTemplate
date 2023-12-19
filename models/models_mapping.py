from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)


class ModelsMapping(Config):
    popular_model_path: str = "model_weights/popular_model.json"
    userknn_model_path: str = "model_weights/userknn_model.pkl"
    als_model_path: str = "model_weights/ALS_4.pkl"
    lfm_logistic_4_model_path: str = "model_weights/LightFM_logistic_4.pkl"
    lfm_warp_4_model_path: str = "model_weights/LightFM_warp_4.pkl"
    lfm_logistic_4_ann_model_path: str = "model_weights/LightFM_logistic_4_ANN.pkl"
    lfm_warp_4_ann_model_path: str = "model_weights/LightFM_warp_4_ANN.pkl"
    multivae_model_path: str = "model_weights/MultiVAE-recommendations.json"
    ae_attention_path: str = "model_weights/custom-ae-attention-recommendations.json"
