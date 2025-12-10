from src.lab05.json_csv import json_to_csv

json_path = "data/samples/people.json"
csv_path = "data/out/people_from_json.csv"

json_to_csv(json_path, csv_path)
print("Конвертация JSON → CSV выполнена.")
