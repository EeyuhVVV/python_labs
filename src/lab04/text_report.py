import sys
import argparse
from pathlib import Path

from lab04.io_txt_csv import read_text, write_csv
from lab03.text import normalize, tokenize, count_freq, top_n


def process_file(path: Path, encoding: str):
    text = read_text(path, encoding=encoding)
    norm = normalize(text, casefold=True, yo2e=True)
    tokens = tokenize(norm)
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="ЛР4 — генерация отчёта по частотам слов"
    )

    parser.add_argument("--in", dest="inputs", nargs="+", default=["data/lab04/input.txt"],
                        help="Входные файлы")
    parser.add_argument("--out", dest="out", default="data/lab04/report.csv",
                        help="Выходной CSV отчёт")
    parser.add_argument("--encoding", default="utf-8", help="Кодировка входных файлов")

    parser.add_argument("--per-file", dest="per_file", default=None,
                        help="CSV отчёт по каждому файлу отдельно (file, word, count)")
    parser.add_argument("--total", dest="total", default=None,
                        help="CSV общий отчёт по всем файлам вместе")

    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]

    if len(inputs) == 1 and args.per_file is None and args.total is None:
        tokens = process_file(inputs[0], args.encoding)

        if not tokens:
            write_csv([], args.out, header=("word", "count"))
            print("Всего слов: 0")
            print("Уникальных слов: 0")
            print("Топ-5:")
            return

        freq = count_freq(tokens)
        sorted_items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

        write_csv(sorted_items, args.out, header=("word", "count"))

        print(f"Всего слов: {len(tokens)}")
        print(f"Уникальных слов: {len(freq)}")

        top5 = top_n(freq, 5)
        print("Топ-5:")
        for w, c in top5:
            print(f"{w}:{c}")

        return

    all_tokens = []
    per_file_rows = []

    for p in inputs:
        tokens = process_file(p, args.encoding)
        all_tokens.extend(tokens)

        freq = count_freq(tokens)
        sorted_items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

        for w, c in sorted_items:
            per_file_rows.append((p.name, w, c))

    if args.per_file:
        write_csv(per_file_rows, args.per_file, header=("file", "word", "count"))

    if args.total:
        freq_total = count_freq(all_tokens)
        sorted_total = sorted(freq_total.items(), key=lambda kv: (-kv[1], kv[0]))
        write_csv(sorted_total, args.total, header=("word", "count"))

if __name__ == "__main__":
    main()
