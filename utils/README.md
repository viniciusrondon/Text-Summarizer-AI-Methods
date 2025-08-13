# CustomYoutubeLoader

## Purpose
This class provides a custom implementation for loading YouTube video transcripts directly using the `youtube-transcript-api` library. It is designed to bypass compatibility issues between older versions of LangChain's `YoutubeLoader` and newer versions of `youtube-transcript-api`. The loader returns a list of LangChain `Document` objects, making it compatible with LangChain chains (e.g., summarization).

It handles language fallbacks, translations, and common errors gracefully.

## Requirements
- `youtube-transcript-api` (version >=1.2.2 recommended)
- `langchain-core` (for Document objects)

## Usage

### Import
```python
from utils.custom_youtube_loader import CustomYoutubeLoader
```

### Parameters
- `video_id` (Optional[str]): The YouTube video ID (e.g., "dQw4w9WgXcQ"). Provide either this or `youtube_url`.
- `youtube_url` (Optional[str]): The full YouTube URL (e.g., "https://www.youtube.com/watch?v=dQw4w9WgXcQ").
- `language` (Union[str, Sequence[str]]): List of preferred language codes (e.g., ["en", "pt"]). Defaults to ["en", "pt", "en-US", "pt-BR"].
- `translation` (Optional[str]): If provided, translate the transcript to this language code (e.g., "en").

### Example
```python
from utils.custom_youtube_loader import CustomYoutubeLoader

# Load from URL with language fallbacks
loader = CustomYoutubeLoader(youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", language=["en", "pt"])
docs = loader.load()

if docs:
    print(docs[0].page_content)  # Prints the full transcript
else:
    print("No transcript available.")
```

### Error Handling
- Returns an empty list if transcripts are disabled, not found, or the video is unavailable.
- Raises RuntimeError for other unexpected errors.

### Notes
- This is a workaround for production stability. If using in a Streamlit app, wrap in try-except to display user-friendly errors.
- For more advanced features, consider contributing to or upgrading LangChain.
