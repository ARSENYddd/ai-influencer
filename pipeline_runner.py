"""
Orchestrates the full pipeline for all 3 personas.
Called by scheduler or manually via CLI: python pipeline_runner.py
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_personas() -> list[dict]:
    personas = []
    for f in Path("config/personas").glob("*.json"):
        try:
            personas.append(json.loads(f.read_text()))
        except Exception as e:
            logger.error(f"Failed to load persona {f}: {e}")
    return personas


def run_pipeline(dry_run: bool = False):
    logger.info("=== Pipeline run started ===")

    from config.settings import settings
    from scraper.reels_scraper import ReelsScraper
    from analyzer.trend_analyzer import TrendAnalyzer
    from planner.content_planner import assign_trends_to_personas
    from caption.caption_generator import CaptionGenerator
    from pipelines.pipeline_c.image_generator import ImageGenerator
    from pipelines.pipeline_c.reels_montage import ReelsMontage
    from database.db import SessionLocal
    from database.models import ContentItem

    db = SessionLocal()

    try:
        personas = load_personas()
        if not personas:
            logger.error("No persona files found in config/personas/")
            return

        # 1. Scrape trends
        logger.info("Scraping trending reels...")
        try:
            scraper = ReelsScraper(
                settings.INSTAGRAM_SCRAPER_USERNAME,
                settings.INSTAGRAM_SCRAPER_PASSWORD
            )
            raw_reels = scraper.get_trending_reels(count_per_tag=10, min_likes=5000)
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            raw_reels = []

        # 2. Analyze trends
        logger.info("Analyzing trends with Groq...")
        analyzer = TrendAnalyzer(settings.GROQ_API_KEY)
        top_trends = analyzer.analyze_trends(raw_reels, personas)

        if not top_trends:
            logger.warning("No trends found from analyzer — using fallback")
            top_trends = [{
                "id": "fallback",
                "dance_type": "aesthetic pose",
                "energy_level": "medium",
                "score": 50,
                "best_persona": personas[0]["id"],
                "pipeline_recommendation": "C",
                "image_prompt_keywords": ["dancing", "aesthetic", "trending"],
                "reason": "fallback trend"
            }]

        # 3. Assign trends to personas
        assignments = assign_trends_to_personas(personas, top_trends)

        # 4. Generate content for each persona
        image_gen = ImageGenerator(settings.REPLICATE_API_TOKEN)
        montage = ReelsMontage()
        caption_gen = CaptionGenerator(settings.GROQ_API_KEY)

        for assignment in assignments:
            persona = assignment["persona"]
            trend = assignment["trend"]
            logger.info(f"Generating content for {persona['id']} | trend: {trend['dance_type']}")

            if dry_run:
                logger.info(f"[DRY RUN] Would generate for {persona['id']}: {trend['dance_type']}")
                continue

            pipeline = trend.get("pipeline_recommendation", settings.PIPELINE_DEFAULT)
            output_dir = f"output/pending/{persona['id']}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            result = None
            image_paths = []

            try:
                if pipeline == "C":
                    image_paths = image_gen.generate(persona, trend, output_dir, count=4)
                    if image_paths:
                        video_path = f"{output_dir}/reel_{trend['id']}.mp4"
                        result = montage.build(image_paths, video_path)
                elif pipeline == "A":
                    logger.info(f"Pipeline A: queuing face swap for {persona['id']} (manual step)")
                    result = None
            except Exception as e:
                logger.error(f"Content generation failed for {persona['id']}: {e}", exc_info=True)
                continue

            # Generate caption (non-fatal)
            caption = None
            try:
                caption = caption_gen.generate(persona, trend)
            except Exception as e:
                logger.error(f"Caption generation failed for {persona['id']}: {e}")

            # Save to DB
            item = ContentItem(
                persona_id=persona["id"],
                trend_id=trend["id"],
                trend_type=trend["dance_type"],
                pipeline=pipeline,
                status="pending",
                video_path=result,
                caption=caption,
                image_paths=json.dumps(image_paths),
                score=trend.get("score", 0),
            )
            db.add(item)
            db.commit()
            logger.info(f"Content item saved for {persona['id']} (DB ID: {item.id})")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        db.close()
        logger.info("=== Pipeline run complete ===")


if __name__ == "__main__":
    import argparse
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            _logging.FileHandler("logs/app.log"),
            _logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Run AI Influencer Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without making API calls")
    args = parser.parse_args()

    from database.db import init_db
    init_db()

    run_pipeline(dry_run=args.dry_run)
