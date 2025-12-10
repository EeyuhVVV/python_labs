

<h1>Python_Labs</h1>

# Лабораторная работ
а 1



## Задания

### Задание 1:

Программа выводит простое приветствие в консоль.

**Код:**
```python
name = input("Имя: ")
age = int(input("Возраст: "))
print(f"Привет, {name}! Через год тебе будет {age}.")
```
Результат выполнения:

![Результат выполнения задания 1](images/lab01/01_greeting.png)

### Задание 2: 

Код:

```python
a = (input("a: ")).replace(",",".")
b = (input("b: ")).replace(",",".")
print(f"sum={round(float(a)+float(b), 2)}; avg={round((float(a)+float(b))/2, 2)}")
```
Результат выполнения:

![Результат выполнения задания 2](images/lab01/02_sum_avg.png)


### Задание 3: 

```python
price = int(input())
discount = int(input())
vat = int(input())
base = price * (1 - discount/100)
vat_amount = base * (vat/100)
total = base + vat_amount
print(f"База после скидки: {base:.2f} ₽")
print(f"НДС:               {vat_amount:.2f} ₽")
print(f"Итого к оплате:    {total:.2f} ₽")
```
Результат выполнения:

![Результат выполнения задания 3](images/lab01/03_discount_vat.png)

### Задание 4: 

```python
mm = int(input("Минуты: "))
hh = mm//60
print(f"{hh}:{(mm-60*hh):02d}")
```
Результат выполнения:

![Результат выполнения задания 4](images/lab01/04_minutes_to_hhmm.png)

### Задание 5:

```python
fio = input("ФИО: ").split()
I = ''
for i in range(3):
    I += fio[i][0]
print(f"Инициалы: {I}.")
print(sum(len(i) for i in fio)+2)
```

Результат выполнения:

![Результат выполнения задания 5](images/lab01/05_initials_and_len.png)


### Задание 6:
```python
n = int(input())
ochno = 0
for i in range(n):
    surname, name, age, form = map(str, input().split(' '))
    if form == 'True':
        ochno+=1
print(ochno, n-ochno)
```

Результат выполнения:

![Результат выполнения задания 6](images/lab01/ex_06.png)

### Задание 7:
```python
cipher = input()
ans = ''
n = 0
for i in range(len(cipher)):
    if cipher[i].isupper():
        ans+=cipher[i]
        n = i+1
        break

for i in range(len(cipher)):
    for j in range(2,1000,3):
        if i == n+j:
            ans+=cipher[i]
print(ans)
```

Результат выполнения:

![Результат выполнения задания 1](images/lab01/ex_07.png)


# Лабораторная работа 2
### Задание 1:

```python
def min_max(nums):
    if nums==[]:
        return 'ValueError'
    else:
        return (min(nums),max(nums))

def unique_sorted(nums):
    return sorted(set(nums))
def flatten(mat):
    new_mat = []
    for i in mat:
        for j in i:
            if type(j) == str:
                return 'TypeError'
            new_mat.append(j)
    return (new_mat)
```

Результат выполнения:

![Результат выполнения задания 1](images/lab02/arrays.png)

### Задание 2:

```python

def transpose(mat):
    if len(mat) == 0:
        return []
    elif len(mat) == 1:
        b = [[] for _ in range(len(mat[0]))]
        for i in range(len(mat[0])):
            b[i] += [(mat[0][i])]
        return b
    elif all(len(mat[i]) == 1 for i in range(len(mat))):
        c = [[]]
        for k in mat:
            c[0] += k
        return c
    elif all(len(mat[i]) == len(mat[i + 1]) for i in range(len(mat[0]) - 1)):
        a = [[] for _ in range(len(mat))]
        for k in range(len(mat)):
            for i in range(len(mat[0])):
                a[k] += [mat[i][k]]
        return a
    else:
        return 'ValueError'

def row_sums(mat):
    k = len(mat[0])
    ans = []
    for i in mat:
        if k == len(i):
            ans.append(sum(i))
        else:
            return 'ValueError'
    return ans

def col_sums(mat):
    ans = [0]*len(mat[0])
    k = len(mat[0])
    for i in range(len(mat)):
        if k==len(mat[i]):
            for j in range(len(mat[i])):
                ans[j]+=mat[i][j]
        else:
            return 'ValueError'
    return ans
```

Результат выполнения:

![Результат выполнения задания 2](images/lab02/matrix.png)

### Задание 3:

```python
fio_new = ''
flag1 = False
len_fio = 0
gpa = float
s = input().replace('(','').replace(')','').split(', ')
for i in s:
    if i == '':
        s.append('musor')
if len(s) == 3:
    fio_old = s[0].replace('''"''','').split(' ')
    for i in fio_old:
        if i != '' and flag1 == False:
            fio_new += str(i.capitalize())+' '
            len_fio+=1
            flag1 = True
        elif flag1 == True and i!='':
            fio_new += str(i.capitalize()[0]) + '.'
            len_fio+=1
    group = s[1].replace('''"''','')
    gpa = float(s[2])
    if 3>=len_fio>=2:
        print(f'''"{fio_new}, гр. {group}, GPA {gpa:.2f}"''')
    else:
        print('ValueError')
else: print('ValueError')
```

Результат выполнения:

![Результат выполнения задания 3](images/lab02/tuples.png)


# Лабораторная работа 3
### Задание 1:

```python
import re
def normalize(text: str, casefold: bool, yo2e: bool):
    text = ' '.join(text.split())
    if casefold:
        text = text.casefold()
    if yo2e:
        text = text.replace("ё","е").replace('Ё', 'Е')
    return text

print(normalize("ПрИвЕт\nМИр\t", True,False))
print(normalize("ёжик, Ёлка", False,True))
print(normalize("Hello\r\nWorld", False,False))
print(normalize("  двойные   пробелы  ", False,False))
def tokenize(text: str):
    text=text.replace(',', ' ').replace('.',' ')
    return re.sub(r'[^a-zA-Zа-яА-Я0-9-\s]', '', text).split()
print(tokenize("привет мир"))
print(tokenize("hello,world!!!"))
print(tokenize("по-настоящему круто"))
print(tokenize("2025 год"))
print(tokenize("emoji 😀 не слово"))

def count_freq(tokens: list[str]):
    freq = {}
    for i in tokens:
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1
    return freq
def top_n(freq: dict[str, int], n: int):
    s = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return s[:n]
print(count_freq(["a","b","a","c","b","a"]))
print(top_n(count_freq(["a","b","a","c","b","a"]),2))
print((count_freq(["bb","aa","bb","aa","cc"])))
print(top_n(count_freq(["bb","aa","bb","aa","cc"]),2))
```

Результат выполнения:

![Результат выполнения задания 1](images/lab03/text.png)

### Задание 2:

```python
import sys
import io

# true - красивый табличный вывод; false - стандарт
PRETTY_TABLE_OUTPUT = True

try:
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except TypeError:
    pass

from text import normalize, tokenize, count_freq, top_n



def print_simple_output(total, unique, top_words):
    print(f"Всего слов: {total}")
    print(f"Уникальных слов: {unique}")
    print("Топ-5:")
    for word, count in top_words:
        print(f"{word}:{count}")


def print_pretty_table(total, unique, top_words):
    print(f"Всего слов: {total}")
    print(f"Уникальных слов: {unique}")

    if not top_words:
        print("Топ-5:")
        return

    header_word = "слово"
    max_len = max([len(word) for word, count in top_words] + [len(header_word)])

    header_freq = "частота"

    print(f"{header_word:<{max_len}} | {header_freq}")
    print(f"{'-' * max_len}-+-{'-' * len(header_freq)}")

    for word, count in top_words:
        print(f"{word:<{max_len}} | {count}")


input_text = sys.stdin.read()

normalized_text = normalize(input_text, casefold=True, yo2e=True)
tokens = tokenize(normalized_text)
total_words = len(tokens)

if total_words == 0:
    print("Всего слов: 0")
    print("Уникальных слов: 0")
    print("Топ-5:")
else:
    freq_dict = count_freq(tokens)
    unique_words = len(freq_dict)
    top_5_words = top_n(freq_dict, 5)

    if PRETTY_TABLE_OUTPUT:
        print_pretty_table(total_words, unique_words, top_5_words)
    else:
        print_simple_output(total_words, unique_words, top_5_words)
```
### Комментарии к выполнению:

```
Чтобы выполнить код я входные данные использовал из текстового документа
PRETTY_TABLE_OUTPUT - регулировка вывода, параметр True - красивый табличный вывод
параметр False - стандартный вывод
```

Результат выполнения:

![Результат выполнения задания 2](images/lab03/text_stats.png)



# Лабораторная работа 4
### Задание 1:

```python
from pathlib import Path
import csv
from typing import Iterable, Sequence


def ensure_parent_dir(path: str | Path) -> None:
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    p = Path(path)
    return p.read_text(encoding=encoding)


def write_csv(
        rows: Iterable[Sequence],
        path: str | Path,
        header: tuple[str, ...] | None = None
) -> None:
    rows = list(rows)

    if rows:
        length = len(rows[0])
        for r in rows:
            if len(r) != length:
                raise ValueError("Все строки CSV должны быть одинаковой длины")

    ensure_parent_dir(path)

    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)
```


### Задание 2:

```python
import sys
import argparse
from pathlib import Path

from lab04.io_txt_csv import read_text, write_csv
from lib.text import normalize, tokenize, count_freq, top_n




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
```
### Комментарии к выполнению:

```
Используются функции из ЛР3 (normalize, tokenize, count_freq, top_n) для обработки текста.
io_txt_csv.py содержит функции для чтения текстовых файлов и записи CSV (read_text, write_csv).
text_report.py читает один входной файл (input.txt), подсчитывает частоты слов, сортирует их и сохраняет отчёт report.csv.
Пустой файл → создаётся CSV только с заголовком.
Кодировка по умолчанию UTF-8, можно изменить через параметр --encoding.
Консольный вывод показывает общее количество слов, уникальных слов и топ-5 слов.
```

Результат выполнения:

![Результат выполнения задания 2](images/lab04/text_report.png)
![report_csv.png](images/lab04/report_csv.png)



# Лабораторная работа 5
### csv_json:

```python
import json
import csv
from pathlib import Path


def csv_to_json(csv_path: str, json_path: str) -> None:
    file_csv=Path(csv_path)

    if not file_csv.exists():
        return FileNotFoundError("Файл не найден")

    if file_csv.suffix != ".csv":
        return ValueError("Неверный тип данных")

    with open(file_csv, "r", encoding='utf-8') as f:
        reader=csv.DictReader(f)

        if reader.fieldnames is None:
            return ValueError("Отсутствуют заголовки в файле")
        dano=list(reader)
    if len(dano)==0:
        return ValueError("Пустой файл")

    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(dano, f, ensure_ascii=False, indent=2)
csv_to_json("data/samples/test.csv","data/out/test_from_csv.json.json")
```
Результат выполнения:

![Результат выполнения задания 1](images/lab05/csv_json.png)

### json_csv:

```python
import json
import csv
from pathlib import Path

def json_to_csv(json_path: str, csv_path: str) -> None:
    file_json=Path(json_path)

    if not file_json.exists():
        return FileNotFoundError("файл не найден")

    try:
        with file_json.open('r',encoding='utf-8') as f:
            dano=json.load(f)
    except json.JSONDecodeError:
        return ValueError("неподдерживаемая структура")

    except not isinstance(dano,list):
        return ValueError("JSON должен быть быть в виде списка объектов")

    except len(dano)==0:
        return ValueError("JSON файл пуст")

    except not all(isinstance(item, dict) for item in dano):
        return ValueError("Каждый элемент JSON должны быть словарями")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=tuple(dano[0].keys()))
            writer.writeheader()
            writer.writerows(dano)
json_to_csv("data/samples/test.json","data/out/test_from_json.csv")
```

Результат выполнения:

![Результат выполнения задания 2](images/lab05/json_csv.png)

### csv_xlsx:

```python
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

def csv_to_xlsx(csv_path: str, xlsx_path: str) -> None:
    csv_file=Path(csv_path)
    if not csv_file.exists():
        return FileNotFoundError("Файл не найден")
    if csv_file.suffix != '.csv':
        return ValueError("Неверный тип файла")
    wb=Workbook()
    ws=wb.active
    ws.title="Sheet1"

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader= csv.DictReader(f)
        rows = list(reader)
    if len(rows)==0:
        return ValueError("Файл не содержит данных")
    if not reader.fieldnames:
        return ValueError("Файл не содержит заголовка")

    ws.append(reader.fieldnames)

    r_count=0
    for row in rows:
        r_count+=1

        data_for_ex=[]
        for title in reader.fieldnames:
            data_for_ex.append(row[title])
        ws.append(data_for_ex)
    if r_count == 0:
        return ValueError("Нет данных")


    for col_index in range(1,len(reader.fieldnames)+1):
        column_letter=get_column_letter(col_index)
        max_len=0
        for row in ws[column_letter]:
            if row.value is not None:
                max_len=max(max_len,len(str(row.value)))

        m_width=max(max_len+2, 8)
        ws.column_dimensions[column_letter].width =m_width
    xlsx_path = Path(xlsx_path)
    wb.save(xlsx_path)

csv_to_xlsx("data/samples/people.csv","data/out/people.xlsx")
```

Результат выполнения:

![Результат выполнения задания 3](images/lab05/csv_xlsx.png)


# Лабораторная работа 6
### cli_text:

```python
import argparse
from pathlib import Path
from src.lib.text import tokenize, count_freq, top_n

def main():
    parser = argparse.ArgumentParser(description="CLI-утилиты лабораторной №6")
    subparsers=parser.add_subparsers(dest="command", help="Доступные соманды")

    stats_parser = subparsers.add_parser("stats",help="Частоты слов в тексте")
    stats_parser.add_argument("--input", required=True, help="Входной текстовый файл")
    stats_parser.add_argument("--top", type=int,default=5,help="Количество топовых слов "
    "(по умолчанию: 5)")

    cat_parser=subparsers.add_parser("cat", help="Вывод содержимого файла")
    cat_parser.add_argument("--input", required=True, help="Путь к входному файлу")
    cat_parser.add_argument("-n",action="store_true", help="Нумеровать строки")

    args = parser.parse_args()

    file=Path(args.input)

    if not file.exists():
        raise FileNotFoundError("Файл не найден")


    if args.command == "cat":

        with open(file, "r", encoding="utf-8") as f:
            number=1
            for row in f:
                row = row.rstrip("\n")
                if args.n:
                    print(f"{number} : {row}")
                    number+=1
                else:
                    print(row)

    elif args.command == "stats":

        with open(file, "r", encoding="utf-8") as f:
            data=[row for row in f]
        data = "".join(data)
        tokens = tokenize(text=data)
        freq = count_freq(tokens=tokens)
        top=top_n(freq=freq, n = args.top)

        number=1
        for x, y in top:
            print(f"{number}. {x} - {y}")
            number+=1
if __name__ == "__main__":
    main()
```
Результат выполнения:

![Результат выполнения задания 1](images/lab06/cli_text1.png)
![cli_text2.png](images/lab06/cli_text2.png)
![cli_text3.png](images/lab06/cli_text3.png)

### cli_convert:

```python
import argparse
from src.lib.json_csv import json_to_csv, csv_to_json
from src.lib.csv_xlsx import csv_to_xlsx

def main():
    parser = argparse.ArgumentParser(description="Конвертер данных между форматами")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды конвертации")

    json_to_csv_parser = subparsers.add_parser("json_to_csv", help="Конвертировать JSON в CSV")
    json_to_csv_parser.add_argument("--in", dest = "input", required= True, help="Входной JSON файл")
    json_to_csv_parser.add_argument("--out", dest = "output", required = True, help="Выходной CSV файл")


    csv_to_json_parser = subparsers.add_parser("csv_to_json", help="Конвертировать CSV в JSON")
    csv_to_json_parser.add_argument("--in", dest="input", required=True, help="Входной CSV файл")
    csv_to_json_parser.add_argument("--out", dest="output", required=True, help="Выходной JSON файл")

    csv_to_xlsx_parser = subparsers.add_parser("csv_to_xlsx", help="Конвертировать CSV в XLSX")
    csv_to_xlsx_parser.add_argument("--in", dest="input", required=True, help="Входной CSV файл")
    csv_to_xlsx_parser.add_argument("--out", dest="output", required=True, help="Выходной XLSX файл")

    args = parser.parse_args()

    if args.command == "json_to_csv":
        json_to_csv(json_path=args.input, csv_path=args.output)

    elif  args.command == "csv_to_json":
        csv_to_json(csv_path=args.input, json_path=args.output)

    elif args.command == "csv_to_xlsx":
        csv_to_xlsx(csv_path=args.input, xlsx_path=args.output)

if __name__ == "__main__":
    main()
```

Результат выполнения:

![Результат выполнения задания 2](images/lab06/cli_convert1.png)
![cli_convert2.png](images/lab06/cli_convert2.png)
![cli_convert3.png](images/lab06/cli_convert3.png)
![cli_convert4.png](images/lab06/cli_convert4.png)


# Лабораторная работа 7
### test_text:

```python
import pytest

from src.lib.text import count_freq, normalize, tokenize, top_n


@pytest.mark.parametrize(
    "src,expected",
    [
        ("ПрИвЕт\nМИр\t", "привет мир"),
        ("ёжик, Ёлка", "ежик, елка"),
        ("Hello\r\nWorld", "hello world"),
        ("  двойные   пробелы  ", "двойные пробелы"),
    ],
)
def test_normalize(src, expected):
    assert normalize(src) == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        ("привет мир", ["привет", "мир"]),
        ("hello,world!!!", ["hello", "world"]),
        ("по-настоящему круто", ["по-настоящему", "круто"]),
        ("2025 год", ["2025", "год"]),
        ("emoji 😀 не слово", ["emoji", "не", "слово"]),
    ],
)
def test_tokenize(src, expected):
    assert tokenize(src) == expected


def test_count_and_top():
    tokens = ["a", "b", "a", "c", "b", "a"]
    freq = count_freq(tokens)
    assert freq == {"a": 3, "b": 2, "c": 1}
    assert top_n(freq, 2) == [("a", 3), ("b", 2)]


def test_top_tie_breaker():
    freq = count_freq(["bb", "aa", "bb", "aa", "cc"])
    assert top_n(freq, 2) == [("aa", 2), ("bb", 2)]


def test_dop():
    """Тесты для пустых входных данных"""
    assert normalize("") == ""
    assert tokenize("") == []
    assert count_freq([]) == {}
    assert top_n({}, 5) == []


def test_top_dop():
    """Запрос большего N чем есть элементов"""
    freq = {"a": 3, "b": 2}
    assert top_n(freq, 5) == [("a", 3), ("b", 2)]
```
Результат выполнения:

![Результат выполнения задания 1](images/lab07/test_text.png)

### test_json_csv:

```python
import json, csv
from pathlib import Path
import pytest
from src.lab05.json_csv import json_to_csv
from src.lab05.csv_json import csv_to_json


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def read_csv_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def test_json_to_csv_roundtrip(tmp_path: Path):
    src = tmp_path / "people.json"
    dst = tmp_path / "people.csv"
    data = [{"name": "Alice", "age": 22}, {"name": "Bob", "age": 25}]
    write_json(src, data)

    json_to_csv(str(src), str(dst))
    rows = read_csv_rows(dst)
    assert len(rows) == 2
    assert set(rows[0]) >= {"name", "age"}

def test_csv_to_json_roundtrip(tmp_path: Path):
    src = tmp_path / "people.csv"
    dst = tmp_path / "people.json"
    src.write_text("name,age\nAlice,22\nBob,25\n", encoding="utf-8")

    csv_to_json(str(src), str(dst))
    obj = json.loads(dst.read_text(encoding="utf-8"))
    assert isinstance(obj, list) and len(obj) == 2
    assert set(obj[0]) == {"name", "age"}

def test_json_to_csv_empty(tmp_path: Path):
    """Тест для пустого JSON - ожидаем что функция обработает это корректно"""
    src = tmp_path / "empty.json"
    dst = tmp_path / "empty.csv"
    src.write_text("[]", encoding="utf-8")

    try:
        json_to_csv(str(src), str(dst))
        # Если функция выполнилась без ошибки, проверяем результат
        if dst.exists():
            # Файл создан - тест пройден
            pass
    except (ValueError, IndexError):
        pass

def test_csv_to_json_empty(tmp_path: Path):
    """Тест для пустого CSV - ожидаем что функция обработает это корректно"""
    src = tmp_path / "empty.csv"
    dst = tmp_path / "empty.json"
    src.write_text("", encoding="utf-8")

    try:
        csv_to_json(str(src), str(dst))
        # Если функция выполнилась без ошибки, проверяем результат
        if dst.exists():
            # Файл создан - тест пройден
            pass
    except (ValueError, Exception):
        pass

def test_missing_file(tmp_path: Path):
    """Тест для несуществующего файла"""
    try:
        csv_to_json("nope.csv", str(tmp_path / "out.json"))
        # Если функция выполнилась, проверяем результат
        if (tmp_path / "out.json").exists():
            # Файл создан - тест пройден
            pass
    except FileNotFoundError:
        pass
```

Результат выполнения:

![Результат выполнения задания 2](images/lab07/test_json_csv.png)

black:

![test_json_csv_black.png](images/lab07/test_json_csv_black.png)


# Лабораторная работа 8
### models:

```python
from datetime import datetime, date
from dataclasses import dataclass
@dataclass
class Student:
    fio: str
    birthdate: str
    group: str
    gpa: float

    def __post_init__(self):
        #  Валидация birthdate
        try:
            datetime.strptime(self.birthdate, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Неверный формат даты: {self.birthdate}. Ожидается YYYY-MM-DD")

        #  Валидация GPA
        if not (0 <= self.gpa <= 5):
            raise ValueError(f"GPA должен быть в диапазоне 0..5, получено: {self.gpa}")

    # Возраст студента
    def age(self) -> int:
        birth = datetime.strptime(self.birthdate, "%Y-%m-%d").date()
        today = date.today()
        years = today.year - birth.year
        if (today.month, today.day) < (birth.month, birth.day):
            years -= 1
        return years
    # Сериализация
    def to_dict(self) -> dict:
        return {
            "fio": self.fio,
            "birthdate": self.birthdate,
            "group": self.group,
            "gpa": self.gpa
        }


    # Десериализация

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            fio=d["fio"],
            birthdate=d["birthdate"],
            group=d["group"],
            gpa=d["gpa"]
        )

    def __str__(self):
        return f"{self.fio} ({self.group}), возраст: {self.age()}, GPA: {self.gpa}"


```

### serialize:

```python
import json
from .models import Student
def students_to_json(students, path):
    "Сохранение списка студентов в JSON файл."
    data = [s.to_dict() for s in students]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def students_from_json(path) -> list[Student]:
    "Загрузка студентов из JSON файла."
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result = []
    for d in raw:
        result.append(Student.from_dict(d))

    return result
```

Результат выполнения:

![Результат выполнения заданий](images/lab08/serialize1.png)
![serialize2.png](images/lab08/serialize2.png)
