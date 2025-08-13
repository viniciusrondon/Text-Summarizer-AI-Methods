####################### load libraries #######################
import validators, streamlit as st
import os

from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from youtube_transcript_api import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from utils.custom_youtube_loader import CustomYoutubeLoader

####################### load env variables #######################
load_dotenv()

## openai
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## huggingface
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

## llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chunk_prompt = """
You are a helpful assistant that can summarize text.
Write a concise summary of the following text.

Text to summarize: {text}

Summarize the text in 120 words in Brazilian Portuguese.
"""

final_prompt = """
Provide the final summary of the entire speech with these important points.
Add a motivational title in theand quote at the end of the summary.

Write a concise summary of the following text.

Text: {text}

Summarize the text in 120 words in Brazilian Portuguese.
"""

chunk_prompt_template = PromptTemplate(template=chunk_prompt, input_variables=["text"])
final_prompt_template = PromptTemplate(template=final_prompt, input_variables=["text"])

#chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True)
chain = load_summarize_chain(llm=llm,chain_type='map_reduce',map_prompt=chunk_prompt_template, combine_prompt=final_prompt_template, verbose=True)

def load_youtube(url: str):
    """
    Load YouTube transcript using custom loader.
    Returns a list[Document].
    """
    try:
        loader = CustomYoutubeLoader(
            youtube_url=url,
            language=["en", "pt", "en-US", "pt-BR"],
            translation=None,
            start_time=start_time_seconds,
            end_time=end_time_seconds
        )
        return loader.load()
    except Exception as e:
        if "transcripts disabled" in str(e).lower():
            st.error("This video has transcripts disabled. Try another URL.")
        elif "no transcript" in str(e).lower():
            st.error("No transcript available for this video (maybe age-restricted/private).")
        elif "unavailable" in str(e).lower():
            st.error("This video is unavailable.")
        else:
            st.error(f"Error loading transcript: {str(e)}")
        return []

def load_generic_url(url: str):
    """
    Load arbitrary web pages. You may need system deps for some MIME types
    if you rely on 'unstructured' for PDFs/images (libmagic, tesseract, poppler).
    """
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        },
    )
    return loader.load()

def split_docs(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=20000)
    return splitter.split_documents(data)

####################### streamlit app #######################
st.set_page_config(page_title='Text Summarizer', page_icon=':memo:', layout='wide')
st.title('Text Summarizer from Youtube and Web')
st.subheader('Enter the URL of the Youtube video or Web page to summarize')

url = st.text_input('Enter the URL of the Youtube video or Web page to summarize',label_visibility='collapsed')

# Add time window inputs (for YouTube only)
col1, col2 = st.columns(2)
start_time_str = col1.text_input("Start time (HH:MM:SS, optional)", value="")
end_time_str = col2.text_input("End time (HH:MM:SS, optional)", value="")

def time_to_seconds(time_str: str) -> Optional[int]:
    if not time_str:
        return None
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        st.error("Invalid time format. Use HH:MM:SS.")
        return None

start_time_seconds = time_to_seconds(start_time_str)
end_time_seconds = time_to_seconds(end_time_str)


if validators.url(url):
    st.success('Valid URL')
else:
    st.error('Invalid URL')


if st.button("Summarize"):
    if not validators.url(url):
        st.error("Invalid URL")
    else:
        st.success("Valid URL")
        with st.spinner("Summarizing..."):
            # 1) load
            if "youtube.com" in url or "youtu.be" in url:
                data = load_youtube(url)
            else:
                data = load_generic_url(url)

            if not data:
                st.stop()

            # 2) split
            docs = split_docs(data)

            if not docs:
                st.error("Could not extract readable text from this URL.")
                st.stop()

            # 3) summarize (most chains expect input_documents=...)
            try:
                result = chain.invoke({"input_documents": docs})
                output_summary = result["output_text"] if isinstance(result, dict) else result
            except Exception as e:
                st.exception(e)
                st.stop()

            st.session_state['output_summary'] = output_summary
            st.session_state['docs'] = docs
            st.session_state['url'] = url

# Display persisted summary if available
if 'output_summary' in st.session_state:
    st.subheader("Summary")
    st.write(st.session_state['output_summary'])

    if st.button("Save Transcription and Summary"):
        try:
            from pytube import YouTube
            import re
            import os

            url = st.session_state['url']
            docs = st.session_state['docs']

            # Try to get title, with fallback to video ID
            try:
                yt = YouTube(url)
                title = yt.title or "unknown_video"
            except Exception:
                # Fallback: extract video ID from URL
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(url)
                if 'v' in parse_qs(parsed.query):
                    video_id = parse_qs(parsed.query)['v'][0]
                else:
                    video_id = parsed.path.split('/')[-1]
                title = f"youtube_{video_id}"

            safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)

            full_transcript = " ".join(d.page_content for d in docs) if docs else "No transcript"

            data_folder = "data"
            os.makedirs(data_folder, exist_ok=True)

            transcript_path = os.path.join(data_folder, f"{safe_title}_transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(full_transcript)

            summary_path = os.path.join(data_folder, f"{safe_title}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(st.session_state['output_summary'])

            st.success(f"Saved to {data_folder}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    with st.expander("Show extracted content (first chunks)", expanded=False):
        for i, d in enumerate(st.session_state['docs'][:2]):
            st.markdown(f"**Chunk {i+1}** — source: `{d.metadata.get('source', 'n/a')}`")
            st.text(d.page_content[:4000])
    









