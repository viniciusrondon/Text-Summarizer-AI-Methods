import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from utils.custom_youtube_loader import CustomYoutubeLoader

# Load env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompts (from app.py)
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

# Chain
chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=chunk_prompt_template, combine_prompt=final_prompt_template, verbose=True)

# Splitter
def split_docs(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(data)

# Test video
url = "https://www.youtube.com/live/0kvGeJZ9SIk"
video_id = "0kvGeJZ9SIk"  # Extracted

print("Loading transcript...")
loader = CustomYoutubeLoader(youtube_url=url)
data = loader.load()

if not data:
    print("No transcript available.")
    exit()

docs = split_docs(data)
full_transcript = " ".join(d.page_content for d in docs)

print("Summarizing...")
result = chain.invoke({"input_documents": docs})
output_summary = result["output_text"] if isinstance(result, dict) else result

# Save to data/
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

transcript_path = os.path.join(data_folder, f"{video_id}_transcript.txt")
with open(transcript_path, "w", encoding="utf-8") as f:
    f.write(full_transcript)

summary_path = os.path.join(data_folder, f"{video_id}_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(output_summary)

print(f"Saved to {data_folder}: {video_id}_transcript.txt and {video_id}_summary.txt")
