from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)


class ModelsMapping(Config):
    popular_model_path: str = "model_weights/popular_model.json"
    userknn_model_path: str = "model_weights/userknn_model.pkl"
