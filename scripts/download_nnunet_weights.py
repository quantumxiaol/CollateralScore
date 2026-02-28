#!/usr/bin/env python3
"""Download nnUNet segmentation weights from Google Drive into modelsweights/."""

from __future__ import annotations

import argparse
import html
import os
import re
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path
import shutil


DEFAULT_BINARY_FILE_ID = "1p4TYVPz_QA0CvuX5HueYVUUYj-wgs5cj"
DEFAULT_MULTI_FILE_ID = "1r14hRPjsc_443_TZbsRSK3xm-ZbfFlK6"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0"


def is_download_response(response: urllib.response.addinfourl) -> bool:
    content_disposition = response.headers.get("Content-Disposition", "")
    content_type = response.headers.get("Content-Type", "")
    if "attachment" in content_disposition.lower():
        return True
    if "text/html" in content_type.lower():
        return False
    return True


def extract_confirm_token(html: str, cookie_jar: CookieJar) -> str | None:
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return cookie.value

    patterns = [
        r'name="confirm"\s+value="([0-9A-Za-z_]+)"',
        r"'confirm'\s*:\s*'([0-9A-Za-z_]+)'",
        r'"confirm"\s*:\s*"([0-9A-Za-z_]+)"',
        r"confirm=([0-9A-Za-z_]+)&",
        r"confirm=([0-9A-Za-z_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return None


def decode_possible_url(raw_value: str) -> str:
    value = html.unescape(raw_value)
    value = value.replace("\\u003d", "=").replace("\\u0026", "&")
    value = value.replace("\\/", "/")
    return value


def extract_download_url(html_text: str, file_id: str) -> str | None:
    direct_match = re.search(r'"downloadUrl":"([^"]+)"', html_text)
    if direct_match:
        return decode_possible_url(direct_match.group(1))

    form_action_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"',
        html_text,
        flags=re.IGNORECASE,
    )
    if not form_action_match:
        form_action_match = re.search(
            r'<form[^>]+action="([^"]+)"[^>]+id="download-form"',
            html_text,
            flags=re.IGNORECASE,
        )
    if form_action_match:
        action = html.unescape(form_action_match.group(1))
        hidden_inputs = re.findall(
            r'<input[^>]+type="hidden"[^>]*>',
            html_text,
            flags=re.IGNORECASE,
        )
        params: dict[str, str] = {}
        for tag in hidden_inputs:
            name_match = re.search(r'name="([^"]+)"', tag, flags=re.IGNORECASE)
            value_match = re.search(r'value="([^"]*)"', tag, flags=re.IGNORECASE)
            if name_match and value_match:
                params[name_match.group(1)] = html.unescape(value_match.group(1))
        params.setdefault("id", file_id)
        params.setdefault("export", "download")
        return f"{action}?{urllib.parse.urlencode(params)}"

    return None


def extract_filename(response: urllib.response.addinfourl, fallback: str) -> str:
    content_disposition = response.headers.get("Content-Disposition", "")
    encoded_match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition)
    if encoded_match:
        return urllib.parse.unquote(encoded_match.group(1))

    plain_match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if plain_match:
        return plain_match.group(1)

    return fallback


def stream_download_to_file(
    response: urllib.response.addinfourl,
    output_path: Path,
    *,
    append: bool = False,
    initial_bytes: int = 0,
) -> None:
    total_bytes_raw = response.headers.get("Content-Length", "")
    try:
        total_bytes = int(total_bytes_raw)
    except ValueError:
        total_bytes = 0
    if append and total_bytes > 0:
        total_bytes += initial_bytes

    downloaded = initial_bytes
    chunk_size = 1024 * 1024
    last_reported_mb = -1
    mode = "ab" if append else "wb"
    with output_path.open(mode) as file_handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file_handle.write(chunk)
            downloaded += len(chunk)

            downloaded_mb = downloaded // (1024 * 1024)
            if downloaded_mb == last_reported_mb:
                continue
            last_reported_mb = downloaded_mb

            if total_bytes > 0:
                pct = (downloaded / total_bytes) * 100.0
                total_mb = total_bytes / (1024 * 1024)
                print(
                    f"\r[progress] {output_path.name}: {downloaded_mb} MiB / {total_mb:.1f} MiB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r[progress] {output_path.name}: {downloaded_mb} MiB",
                    end="",
                    flush=True,
                )
    print()


def save_download_response(
    opener: urllib.request.OpenerDirector,
    url: str,
    headers: dict[str, str],
    response: urllib.response.addinfourl,
    output_path: Path,
) -> Path:
    existing_size = output_path.stat().st_size if output_path.exists() else 0
    if existing_size <= 0:
        stream_download_to_file(response, output_path)
        response.close()
        return output_path

    response.close()
    range_headers = dict(headers)
    range_headers["Range"] = f"bytes={existing_size}-"
    try:
        resumed_response = opener.open(urllib.request.Request(url, headers=range_headers))
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            print(f"[resume] {output_path.name}: already complete ({existing_size // (1024 * 1024)} MiB)")
            return output_path
        raise

    status_code = resumed_response.getcode() or 200
    if status_code == 206:
        print(f"[resume] {output_path.name}: resuming from {existing_size // (1024 * 1024)} MiB")
        stream_download_to_file(
            resumed_response,
            output_path,
            append=True,
            initial_bytes=existing_size,
        )
        resumed_response.close()
        return output_path

    print(f"[resume] {output_path.name}: server ignored range request, restarting full download")
    stream_download_to_file(resumed_response, output_path)
    resumed_response.close()
    return output_path


def download_from_google_drive(
    file_id: str,
    output_dir: Path,
    fallback_filename: str,
) -> Path:
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    headers = {"User-Agent": USER_AGENT}

    initial_urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
    ]
    attempts: list[str] = []

    for base_url in initial_urls:
        attempts.append(base_url)
        response = opener.open(urllib.request.Request(base_url, headers=headers))
        if is_download_response(response):
            filename = extract_filename(response, fallback_filename)
            output_path = output_dir / filename
            return save_download_response(opener, base_url, headers, response, output_path)

        html_text = response.read().decode("utf-8", errors="ignore")
        response.close()

        parsed_url = extract_download_url(html_text, file_id)
        if parsed_url:
            attempts.append(parsed_url)
            response2 = opener.open(urllib.request.Request(parsed_url, headers=headers))
            if is_download_response(response2):
                filename = extract_filename(response2, fallback_filename)
                output_path = output_dir / filename
                return save_download_response(opener, parsed_url, headers, response2, output_path)
            response2.close()

        confirm_token = extract_confirm_token(html_text, cookie_jar)
        if confirm_token:
            confirm_urls = [
                f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}",
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_token}",
            ]
            for confirm_url in confirm_urls:
                attempts.append(confirm_url)
                response3 = opener.open(urllib.request.Request(confirm_url, headers=headers))
                if is_download_response(response3):
                    filename = extract_filename(response3, fallback_filename)
                    output_path = output_dir / filename
                    return save_download_response(opener, confirm_url, headers, response3, output_path)
                response3.close()

    attempts_text = "\n".join(f"- {url}" for url in attempts)
    raise RuntimeError(
        "Failed to download from Google Drive. The file may require manual access/permission.\n"
        f"File id: {file_id}\n"
        f"Tried URLs:\n{attempts_text}"
    )


def find_nnunet_model_dir(search_root: Path) -> Path | None:
    candidates: list[Path] = []
    for root, dirs, files in os.walk(search_root):
        has_fold = any(dirname.startswith("fold_") for dirname in dirs)
        has_plans = "plans.json" in files or "dataset.json" in files
        if has_fold and has_plans:
            candidates.append(Path(root))
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def flatten_model_dir_if_needed(target_dir: Path) -> Path | None:
    model_dir = find_nnunet_model_dir(target_dir)
    if model_dir is None:
        return None
    if model_dir == target_dir:
        return model_dir

    destination_conflicts = []
    for child in model_dir.iterdir():
        if (target_dir / child.name).exists():
            destination_conflicts.append(child.name)
    if destination_conflicts:
        print(
            "[flatten] skipped due to conflicts in target dir: "
            + ", ".join(sorted(destination_conflicts))
        )
        return model_dir

    print(f"[flatten] {model_dir} -> {target_dir}")
    for child in model_dir.iterdir():
        shutil.move(str(child), str(target_dir / child.name))

    parent = model_dir
    while parent != target_dir:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent

    return find_nnunet_model_dir(target_dir)


def extract_archive_if_supported(archive_path: Path, target_dir: Path) -> bool:
    if zipfile.is_zipfile(archive_path):
        print(f"[extract] {archive_path} -> {target_dir}")
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(target_dir)
        return True

    if tarfile.is_tarfile(archive_path):
        print(f"[extract] {archive_path} -> {target_dir}")
        with tarfile.open(archive_path, "r:*") as tar_file:
            tar_file.extractall(target_dir)
        return True

    return False


def find_candidate_archives(target_dir: Path) -> list[Path]:
    patterns = ["*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2", "*.tar.xz"]
    archives: list[Path] = []
    for pattern in patterns:
        archives.extend(target_dir.glob(pattern))
    return sorted(set(archives), key=lambda path: path.name)


def format_env_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download nnUNet binary/multi model weights into modelsweights/."
    )
    parser.add_argument(
        "--dest-root",
        default="modelsweights",
        help="Destination root directory for downloaded weights (default: modelsweights).",
    )
    parser.add_argument(
        "--binary-id",
        default=DEFAULT_BINARY_FILE_ID,
        help="Google Drive file id for binary nnUNet weights.",
    )
    parser.add_argument(
        "--multi-id",
        default=DEFAULT_MULTI_FILE_ID,
        help="Google Drive file id for multi-label nnUNet weights.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if a valid nnUNet model folder already exists.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not auto-extract zip files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dest_root = Path(args.dest_root).expanduser()
    if not dest_root.is_absolute():
        dest_root = project_root / dest_root
    dest_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("binary", args.binary_id, dest_root / "nnunet" / "binary", "binary_model.zip"),
        ("multi", args.multi_id, dest_root / "nnunet" / "multi", "multi_model.zip"),
    ]

    resolved_model_dirs: dict[str, Path] = {}
    for label, file_id, target_dir, fallback_filename in jobs:
        target_dir.mkdir(parents=True, exist_ok=True)

        flattened_existing = flatten_model_dir_if_needed(target_dir)
        if flattened_existing is not None:
            print(f"[ready] {label}: {flattened_existing}")
            resolved_model_dirs[label] = flattened_existing
            continue

        if not args.no_extract:
            local_archives = find_candidate_archives(target_dir)
            if local_archives:
                print(f"[reuse] {label}: found local archive(s), trying extraction first")
                extracted_any = False
                for local_archive in local_archives:
                    if extract_archive_if_supported(local_archive, target_dir):
                        local_archive.unlink(missing_ok=True)
                        extracted_any = True
                if extracted_any:
                    extracted_model_dir = flatten_model_dir_if_needed(target_dir)
                    if extracted_model_dir:
                        resolved_model_dirs[label] = extracted_model_dir
                        print(f"[ready] {label}: {extracted_model_dir}")
                        continue

        print(f"[download] {label}: id={file_id}")
        archive_or_file = download_from_google_drive(file_id, target_dir, fallback_filename)
        print(f"[saved] {archive_or_file}")

        if not args.no_extract and extract_archive_if_supported(archive_or_file, target_dir):
            archive_or_file.unlink(missing_ok=True)

        resolved_model_dir = flatten_model_dir_if_needed(target_dir)
        if resolved_model_dir:
            resolved_model_dirs[label] = resolved_model_dir
            print(f"[ready] {label}: {resolved_model_dir}")
        else:
            resolved_model_dirs[label] = target_dir
            print(
                f"[warn] {label}: nnUNet model folder not auto-detected under {target_dir}. "
                "Set NNUNET_*_MODEL_DIR manually."
            )

    print("\nSuggested .env entries:")
    print(f"COLLATERALSCORE_MODELS_DIR={format_env_path(dest_root, project_root)}")
    print(
        "NNUNET_BINARY_MODEL_DIR="
        f"{format_env_path(resolved_model_dirs.get('binary', dest_root / 'nnunet' / 'binary'), project_root)}"
    )
    print(
        "NNUNET_MULTI_MODEL_DIR="
        f"{format_env_path(resolved_model_dirs.get('multi', dest_root / 'nnunet' / 'multi'), project_root)}"
    )


if __name__ == "__main__":
    main()
