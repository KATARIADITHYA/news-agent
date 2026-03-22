"""
run_custom.py
─────────────
Run the PolitiSense agent with a custom news text.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_agent, report_to_text

news = "Trump announces 25% tariff on India and unspecified penalties for buying Russian oil"

print(f"\n{'='*65}")
print(f"  Running pipeline for:")
print(f"  \"{news}\"")
print(f"{'='*65}\n")

report = run_agent(news_text=news, thread_id="custom")
print("\n" + report_to_text(report))
