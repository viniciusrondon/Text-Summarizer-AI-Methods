from typing import List, Union, Sequence, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.youtube import _parse_video_id  # Reuse parser

class CustomYoutubeLoader:
    """Custom YouTube transcript loader using youtube-transcript-api directly."""

    def __init__(
        self,
        video_id: Optional[str] = None,
        youtube_url: Optional[str] = None,
        language: Union[str, Sequence[str]] = ["en", "pt", "en-US", "pt-BR"],
        translation: Optional[str] = None,
        start_time: Optional[int] = None,  # in seconds
        end_time: Optional[int] = None,    # in seconds
    ):
        if video_id is None and youtube_url is None:
            raise ValueError("Must provide either video_id or youtube_url")
        if video_id is None:
            video_id = self.extract_video_id(youtube_url)
        self.video_id = video_id
        self.language = [language] if isinstance(language, str) else language
        self.translation = translation
        self.start_time = start_time
        self.end_time = end_time
        self.metadata = {"source": f"https://www.youtube.com/watch?v={video_id}"}

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        video_id = _parse_video_id(youtube_url)
        if not video_id:
            raise ValueError(f"Could not parse video ID from URL: {youtube_url}")
        return video_id

    def load(self) -> List[Document]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

            api = YouTubeTranscriptApi()
            transcript_list = api.list(self.video_id)

            try:
                transcript = transcript_list.find_generated_transcript(self.language)
            except NoTranscriptFound:
                transcript = transcript_list.find_generated_transcript(["en"])

            if self.translation:
                transcript = transcript.translate(self.translation)

            transcript_pieces = transcript.fetch()

            # Filter by time window if provided
            if self.start_time is not None or self.end_time is not None:
                filtered_pieces = []
                for piece in transcript_pieces:
                    piece_start = piece.start
                    piece_end = piece_start + piece.duration
                    if (self.start_time is None or piece_start >= self.start_time) and \
                       (self.end_time is None or piece_end <= self.end_time):
                        filtered_pieces.append(piece)
                transcript_pieces = filtered_pieces

            if not transcript_pieces:
                raise NoTranscriptFound("No transcript pieces in the specified time window.")

            full_transcript = " ".join(piece.text.strip() for piece in transcript_pieces)

            return [Document(page_content=full_transcript, metadata=self.metadata)]

        except TranscriptsDisabled:
            return []
        except NoTranscriptFound:
            return []
        except VideoUnavailable:
            return []
        except Exception as e:
            raise RuntimeError(f"Error loading transcript: {str(e)}") from e
