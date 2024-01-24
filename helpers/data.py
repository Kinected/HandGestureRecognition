def write_labels(path: str, unique_labels: list[str]):
    # Write unique labels to file
    with open(path + "./labels.txt", "w") as f:
        f.write(", ".join(unique_labels))
