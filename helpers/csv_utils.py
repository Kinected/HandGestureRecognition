import csv


def create_csv(path, columns):
    with open(path, "w+", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)


def write_csv(path: str, data: list):
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([data])
