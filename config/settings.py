from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    REPLICATE_API_TOKEN: str = ""
    GROQ_API_KEY: str = ""
    INSTAGRAM_SCRAPER_USERNAME: str = ""
    INSTAGRAM_SCRAPER_PASSWORD: str = ""
    DASHBOARD_PORT: int = 8080
    DASHBOARD_SECRET_KEY: str = "change_this_secret"
    PIPELINE_DEFAULT: str = "C"
    SCRAPE_INTERVAL_HOURS: int = 6
    CONTENT_SCHEDULE_TIMES: str = "09:00,18:00"
    FACEFUSION_HOST: str = "localhost"
    FACEFUSION_PORT: int = 7860

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
