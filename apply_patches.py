import csv
import logging
import shutil
import site
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_patches():
    patches_dir = Path(__file__).parent / "patches"
    valid_suffixes = {".py", ".head", ".tail", ".repl"}
    site_packages_dir = Path(site.getsitepackages()[0])
    for path_to_source_file in patches_dir.rglob("*"):
        suffix = path_to_source_file.suffix
        if suffix not in valid_suffixes:
            continue

        rel_path_to_source_file = path_to_source_file.relative_to(patches_dir)
        src = path_to_source_file
        dst_package = site_packages_dir / rel_path_to_source_file.parents[-2]
        if not dst_package.exists():
            logger.info(
                f"Package `{dst_package.name}` is not installed, skipping patch `{src}`"
            )
            continue
        if suffix in [".head", ".tail"]:
            dst = site_packages_dir / rel_path_to_source_file.with_suffix("")
            if suffix == ".head":
                logger.info(f"Prepending {src} to {dst}")
                with dst.open("r+") as f:
                    content = f.read()
                    f.seek(0)
                    f.write(src.read_text() + "\n" + content)
                    f.truncate()
            else:  # suffix == ".tail"
                logger.info(f"Appending {src} to {dst}")
                with dst.open("a") as f:
                    f.write("\n" + src.read_text())
        elif suffix == ".py":
            dst = site_packages_dir / rel_path_to_source_file
            logger.info(f"Replacing {dst} with {src}")
            shutil.copy(src, dst)
        elif suffix == ".repl":
            dst = site_packages_dir / rel_path_to_source_file.with_suffix("")
            with path_to_source_file.open(
                mode="r", newline="", encoding="utf-8"
            ) as file:
                reader = csv.reader(file, delimiter=";", quotechar='"')
                for row in reader:
                    logger.info(f"Editing {dst} ...")
                    original, replacement = row
                    content = dst.read_text()
                    updated_content = content.replace(original, replacement)
                    dst.write_text(updated_content)


if __name__ == "__main__":
    apply_patches()
