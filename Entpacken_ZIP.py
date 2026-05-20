from pathlib import Path
import zipfile


SOURCE_DIR = Path(r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data")
EXTRACTED_DIR = SOURCE_DIR / "extracted"


def extract_zip_files(source_dir: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(source_dir.glob("*.zip"))

    if not zip_files:
        print(f"Keine ZIP-Dateien gefunden in: {source_dir}")
        return

    for zip_path in zip_files:
        case_name = zip_path.stem
        target_dir = extracted_dir / case_name
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extrahiere: {zip_path.name} -> {target_dir}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir():
                    continue

                filename = Path(member.filename).name

                if not filename.lower().endswith(".xml"):
                    print(f"  Übersprungen, keine XML-Datei: {member.filename}")
                    continue

                target_file = target_dir / filename

                with zip_ref.open(member) as source, open(target_file, "wb") as target:
                    target.write(source.read())

                print(f"  XML extrahiert: {filename}")

    print("Fertig.")


if __name__ == "__main__":
    extract_zip_files(SOURCE_DIR, EXTRACTED_DIR)