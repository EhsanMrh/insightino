import argparse
import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from enums.enum import Directories
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.analysis_pipeline import AnalysisPipeline


def render_pdf(text: str, out_path: str) -> None:
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    for para in text.split("\n\n"):
        story.append(Paragraph(para.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report from a prompt template")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(Directories.MEMORY_STORE.value, exist_ok=True)
    os.makedirs(Directories.REPORTS.value, exist_ok=True)
    os.makedirs(Directories.LOGS.value, exist_ok=True)

    log = init_logger(log_dir=Directories.LOGS.value)
    store = MemoryStore(base_dir=Directories.MEMORY_STORE.value)
    ap = AnalysisPipeline(store=store, log=log)

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    result = ap.answer(namespace=args.namespace, question=prompt_text)
    out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    render_pdf(result["answer"], out_path)  # type: ignore
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()


