from enum import StrEnum
from os import environ
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

class SupaBase (StrEnum):
    URL = st.secrets["SUPABASE_URL"]
    SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
    TABLE_NAME = environ.get('SUPABASE_TABLE_NAME', default='documents') 
    QUERY_NAME = environ.get('SUPABASE_QUERY_NAME', default='match_documents')


class OllamaSettings(StrEnum):
    URL = "localhost:11434"
    EMBEDDINGS_MODEL = "nomic-embed-text"
    CHAT_MODEL = "phi4"



