from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

class MultiLingualSearch:
    def __init__(self):
        self.search = TavilySearchResults(max_results=5)
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

    def translate_query(self, query: str, target_lang: str):
        prompt = f"Translate the following search query into {target_lang}. Only provide the translated text: {query}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Smart extraction for complex Gemini responses
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            content = " ".join(text_parts)
        
        return content.strip()

    def perform_search(self, query: str, languages=["English", "Japanese", "German"]):
        all_results = []
        for lang in languages:
            search_query = query
            if lang != "English":
                search_query = self.translate_query(query, lang)
            
            print(f"Searching in {lang}: {search_query}")
            results = self.search.invoke({"query": search_query})
            for r in results:
                if isinstance(r, dict):
                    r['language'] = lang
                    all_results.append(r)
                else:
                    all_results.append({"content": str(r), "language": lang, "url": "N/A"})
            
        return all_results
