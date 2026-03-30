# AI Influencer Automation

Auto-generate realistic AI girl photos + captions and post to Instagram daily.

## Stack

| Component | Tool |
|-----------|------|
| Image generation | Replicate Flux Dev |
| Caption generation | Anthropic Claude Haiku |
| Instagram posting | instagrapi |
| Scheduling | schedule |

## Quick Start

```bash
# 1. Setup
bash setup.sh

# 2. Fill in credentials
cp .env.template .env
nano .env

# 3. Test without API calls
python pipeline.py --dry-run

# 4. Run live
python pipeline.py

# 5. Start scheduler (09:00 + 18:00 daily)
python scheduler.py
```

## Project Structure

```
ai-influencer/
├── pipeline.py          # Main pipeline orchestrator
├── generator.py         # Image generation (Replicate Flux Dev)
├── caption_generator.py # Caption generation (Claude Haiku)
├── poster.py            # Instagram posting (instagrapi)
├── scheduler.py         # Daily scheduler (09:00 + 18:00)
├── setup.sh             # One-command setup script
├── requirements.txt     # Python dependencies
├── .env.template        # Credentials template
└── output/
    ├── images/          # Generated images (git-ignored)
    └── logs/            # Pipeline logs (git-ignored)
```

## Environment Variables

Copy `.env.template` to `.env` and fill in:

```env
REPLICATE_API_TOKEN=...   # https://replicate.com/account/api-tokens
ANTHROPIC_API_KEY=...     # https://console.anthropic.com/
INSTAGRAM_USERNAME=...    # Your Instagram username
INSTAGRAM_PASSWORD=...    # Your Instagram password
```

## Error Handling

All modules use `tenacity` for automatic retries:
- **Image generation**: 3 retries with exponential backoff (4–30s)
- **Caption generation**: 3 retries with exponential backoff (4–30s)
- **Instagram posting**: 3 retries with exponential backoff (10–60s)

## Scheduler

Posts automatically at **09:00** and **18:00** every day.

```bash
# Run with immediate first post
python scheduler.py --run-now

# Dry-run scheduler (no real API calls)
python scheduler.py --dry-run
```

## Notes

- Instagram session is cached in `session_instagram.json` (git-ignored)
- Images are saved locally in `output/images/` before posting
- All activity is logged to `output/logs/`
