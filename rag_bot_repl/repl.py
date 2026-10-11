"""Console interface for the RAG bot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_bot import RAGBot


SCRIPT_DIR = Path(__file__).resolve().parent
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "выход"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Neon Rift RAG console bot.")
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, etc.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-relevance", type=float, default=0.5)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=SCRIPT_DIR / ".env",
        help="Path to the file with YandexGPT and Chroma settings.",
    )
    return parser.parse_args()


def print_retrieval_debug(bot_response) -> None:
    if not bot_response.chunks:
        print("\n[retrieval] Подходящие чанки не найдены; YandexGPT не вызывался.")
        return

    print("\n[retrieval]")
    for chunk in bot_response.chunks:
        title = chunk.metadata.get("title", "unknown")
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", "unknown")
        print(
            f"- {chunk.relevance:.4f} | {title} | {source} | chunk={chunk_id}"
        )


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    try:
        bot = RAGBot.from_environment(
            device=args.device,
            top_k=args.top_k,
            min_relevance=args.min_relevance,
        )
    except Exception as error:
        print(f"Не удалось запустить бота: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Neon Rift RAG-бот запущен.")
    print("Введите вопрос или /exit для выхода.")
    print(
        f"Параметры retrieval: top_k={args.top_k}, "
        f"min_relevance={args.min_relevance:.2f}"
    )

    while True:
        try:
            query = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nБот: До свидания.")
            break

        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            print("Бот: До свидания.")
            break

        try:
            response = bot.ask(query)
        except Exception as error:
            print(f"Бот: Ошибка при обработке запроса: {error}")
            continue

        print(f"\nБот:\n{response.answer}")
        print_retrieval_debug(response)


if __name__ == "__main__":
    main()
