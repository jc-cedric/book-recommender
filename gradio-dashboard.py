import pandas as pd
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from dotenv import load_dotenv


from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('data/books_with_emotions.csv')

#getting a large thumbnail for the book
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "./data/cover-not-found.jpg",
    books["large_thumbnail"],
)

# reading the tagged description into the TextLoader
raw_documents = TextLoader("./data/tagged_description.txt").load()

# initiating a text_splitter which has as separator a new line
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)

#applying the text_splitter to each documents so we will end up with document chunks that are individual books description
documents = text_splitter.split_documents(raw_documents)

#converting those to documents into embedding using OpenAIEmbeddings and storing them to Chroma vector database
#db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
#huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db_books = Chroma.from_documents(
    documents,
    embedding=hf
)


#applying filtering based on categories and sorting based on emotional tone
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_category"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by=["joy"], ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by=["surprise"], ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by=["anger"], ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by=["fear"], ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by=["sadness"], ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str = None,
        tone: str = None,
):
    recommandations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommandations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) > 1:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{','.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.theme.Glass()) as dashboard:
    gr.Markdown("#Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a book description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button(label="Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)


if __name__ == "__main__":
    dashboard.launch()




