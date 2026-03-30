from dotenv import load_dotenv
load_dotenv()

from solution import RankingEngine, _fmt_result

engine = RankingEngine()

query = input("Search for companies: ").strip()
if not query:
    print("No query entered.")
else:
    results = engine.rank(query)
    if not results:
        print("No results found.")
    else:
        print(_fmt_result(results[0]))
