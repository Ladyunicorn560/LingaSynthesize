from concurrent.futures import ThreadPoolExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

class MultiLingualSearch:
    def __init__(self):
        self.search = TavilySearchResults(max_results=5)
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

    def translate_query(self, query: str, target_lang: str):
        try:
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
            
            translated = content.strip()
            return translated if translated else query
        except Exception as e:
            print(f"Translation fallback for {target_lang}: {e}")
            return query

    def _search_single_language(self, query: str, lang: str):
        search_query = query
        if lang != "English":
            try:
                search_query = self.translate_query(query, lang)
            except Exception as e:
                print(f"Translation warning for {lang}: {e}")
        
        try:
            print(f"Searching in {lang}: {search_query}".encode('utf-8', 'replace').decode('utf-8'))
        except Exception:
            print(f"Searching in {lang}")
        lang_results = []
        try:
            results = self.search.invoke({"query": search_query})
            for r in results:
                if isinstance(r, dict):
                    r['language'] = lang
                    lang_results.append(r)
                else:
                    lang_results.append({"content": str(r), "language": lang, "url": "N/A"})
        except Exception as e:
            print(f"Search warning for {lang}: {e}")
        return lang_results

    def perform_search(self, query: str, languages=["English", "Japanese", "German"]):
        all_results = []
        with ThreadPoolExecutor(max_workers=len(languages)) as executor:
            futures = [executor.submit(self._search_single_language, query, lang) for lang in languages]
            for future in futures:
                all_results.extend(future.result())
            
        return all_results
