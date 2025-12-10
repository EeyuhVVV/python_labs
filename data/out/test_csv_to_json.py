from src.lab05.json_csv import csv_to_json

csv_path = "data/samples/people.csv"
json_path = "data/out/people_from_csv.json"

csv_to_json(csv_path, json_path)
print("Конвертация CSV → JSON выполнена.")
