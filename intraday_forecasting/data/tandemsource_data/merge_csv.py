"""Unisce i CSV nelle sottocartelle e ordina le osservazioni per data."""

import argparse
import csv
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DATE_COLUMNS = (
    "Data e ora dell'evento",
    "Data/Ora di completamento",
    "Data dell'evento",
)


def parse_date(value):
    """Legge date ISO dei CSV Tandem oppure date italiane."""
    value = value.strip()
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        for pattern in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
            try:
                result = datetime.strptime(value, pattern)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Data vuota o non riconosciuta: {value!r}") from None
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def file_type(path):
    """Elimina il prefisso della cartella, delle date o dell'export Tandem."""
    prefix = path.parent.name + "-"
    if path.stem.startswith(prefix):
        return path.stem[len(prefix):]
    match = re.fullmatch(r"(?:\d{8}-\d{8}|[^-]+)-(.+)", path.stem)
    if not match:
        raise ValueError(f"Nome CSV non riconosciuto: {path}")
    return match.group(1)


def merge(root):
    root = Path(root).resolve(strict=True)
    output = root / "merged"
    if output.is_symlink():
        raise ValueError("La cartella merged non deve essere un collegamento.")
    groups = {}
    folders = sorted(
        (p for p in root.iterdir()
         if p.is_dir() and p.name.casefold() != "merged"
         and not p.name.startswith((".", "__")) and not p.is_symlink()),
        key=lambda p: p.name.casefold(),
    )
    for folder in folders:
        for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            if path.is_symlink():
                raise ValueError(f"Il CSV non deve essere un collegamento: {path}")
            kind = file_type(path)
            key = kind.casefold()
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                sample = stream.readline()
                stream.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error as error:
                    raise ValueError(f"Intestazione CSV non riconosciuta: {path}") from error
                reader = csv.DictReader(stream, dialect=dialect, strict=True)
                headers = reader.fieldnames
                if not headers or len(set(headers)) != len(headers) or any(not h for h in headers):
                    raise ValueError(f"Intestazioni vuote o duplicate: {path}")
                candidates = [column for column in DATE_COLUMNS if column in headers]
                if len(candidates) != 1:
                    raise ValueError(f"Colonna della data assente o ambigua: {path}")
                date_column = candidates[0]
                group = groups.setdefault(key, {
                    "name": kind, "headers": headers, "date_column": date_column,
                    "rows": [], "files": 0,
                })
                if set(group["headers"]) != set(headers) or group["date_column"] != date_column:
                    raise ValueError(f"Colonne diverse negli export dello stesso tipo: {path}")
                group["files"] += 1
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Numero di colonne errato: {path}, riga {reader.line_num}")
                    try:
                        timestamp = parse_date(row[date_column])
                    except ValueError as error:
                        raise ValueError(f"{path}, riga {reader.line_num}: {error}") from error
                    group["rows"].append((timestamp, row))
    if not groups:
        raise ValueError(f"Nessun CSV trovato nelle sottocartelle di {root}")

    # Valida tutte le sorgenti prima di scrivere. Ogni risultato viene poi
    # sostituito con un file completo, senza aggiungere righe al vecchio CSV.
    with tempfile.TemporaryDirectory(prefix=".merged-", dir=root) as staging:
        for key, group in sorted(groups.items()):
            group["rows"].sort(key=lambda item: item[0])
            name = f"merged-{group['name']}.csv"
            with (Path(staging) / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=group["headers"])
                writer.writeheader()
                writer.writerows(row for _, row in group["rows"])
        output.mkdir(exist_ok=True)
        for group in groups.values():
            name = f"merged-{group['name']}.csv"
            os.replace(Path(staging) / name, output / name)

    for group in groups.values():
        print(f"merged-{group['name']}.csv: {len(group['rows'])} righe, {group['files']} file sorgente")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Cartella esterna (predefinita: quella dello script)")
    args = parser.parse_args()
    try:
        merge(args.root)
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERRORE: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
