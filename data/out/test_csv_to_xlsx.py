from src.lab05.csv_xlsx import csv_to_xlsx

csv_path = "data/samples/cities.csv"
xlsx_path = "data/out/cities.xlsx"

csv_to_xlsx(csv_path, xlsx_path)
print("Конвертация CSV → XLSX выполнена.")
