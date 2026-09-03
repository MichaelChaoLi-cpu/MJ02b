#!/usr/bin/env python3
"""Inventory core CSES modules stored in ZIP and nested-ZIP archives.

The generic MiliFrame variable scanner only sees unpacked tabular files. CSES
is distributed primarily as Stata files inside archives, so this script reads
archive members in memory and writes the database-build variable
list without modifying ``data/raw``.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd


MAX_ARCHIVE_DEPTH = 3
SAMPLE_ROWS = 1_000
WAVE_PATTERN = re.compile(r"20(?:04|07|09|11|12|13|14|16|17|19|21)")

MODULE_TITLES = {
    "household_core": "Household Core and Survey Identifiers",
    "household_members": "Household Members and Demographics",
    "household_weights": "Household Weights",
    "person_weights": "Person Weights",
    "education": "Education",
    "employment_current": "Current Employment",
    "employment_usual": "Usual Employment",
    "migration": "Migration",
    "food_consumption": "Food Consumption",
    "nonfood_consumption": "Non-food Consumption",
    "food_security": "Food Security",
    "vulnerability": "Household Vulnerability",
    "housing": "Housing",
    "agriculture_land": "Agricultural Land",
    "agriculture_crop_production": "Crop Production",
    "agriculture_crop_costs": "Crop Inputs and Costs",
    "agriculture_crop_sales": "Crop Sales",
}


@dataclass(frozen=True)
class DataSource:
    root_file: Path
    archive_members: tuple[str, ...] = ()

    def display_name(self, root: Path) -> str:
        label = str(self.root_file.relative_to(root))
        if self.archive_members:
            label += "::" + "::".join(self.archive_members)
        return label

    def read_bytes(self) -> bytes:
        payload: bytes | None = None
        for member in self.archive_members:
            archive_input = self.root_file if payload is None else io.BytesIO(payload)
            with zipfile.ZipFile(archive_input) as archive:
                payload = archive.read(member)
        if payload is None:
            raise ValueError(f"No archive payload for {self.root_file}")
        return payload


def is_noise(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return any(
        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
        for part in parts
    )


def discover_archive(
    archive_path: Path,
    members: tuple[str, ...] = (),
    archive_bytes: bytes | None = None,
    depth: int = 0,
) -> list[DataSource]:
    if depth > MAX_ARCHIVE_DEPTH:
        return []
    archive_input = archive_path if archive_bytes is None else io.BytesIO(archive_bytes)
    found: list[DataSource] = []
    try:
        with zipfile.ZipFile(archive_input) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or is_noise(info.filename):
                    continue
                suffix = PurePosixPath(info.filename).suffix.lower()
                chain = members + (info.filename,)
                if suffix == ".dta":
                    found.append(DataSource(archive_path, chain))
                elif suffix == ".zip":
                    try:
                        nested = archive.read(info)
                    except Exception:
                        continue
                    found.extend(
                        discover_archive(archive_path, chain, nested, depth + 1)
                    )
    except (OSError, zipfile.BadZipFile):
        return []
    return found


def discover_sources(root: Path) -> list[DataSource]:
    raw = root / "data" / "raw" / "CSE"
    sources = [DataSource(path) for path in raw.rglob("*.dta")]
    for archive in raw.rglob("*.zip"):
        sources.extend(discover_archive(archive))
    unique = {source.display_name(root): source for source in sources}
    return [unique[name] for name in sorted(unique)]


def normalize_wave(source_name: str) -> str:
    matches = WAVE_PATTERN.findall(source_name)
    if not matches:
        return "unknown"
    return {
        "2004": "2004",
        "2007": "2007",
        "2009": "2009",
        "2011": "2011-12",
        "2012": "2011-12",
        "2013": "2013",
        "2014": "2014",
        "2016": "2016",
        "2017": "2017",
        "2019": "2019",
        "2021": "2021",
    }[matches[0]]


def token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def source_leaf(source: DataSource) -> str:
    target = source.archive_members[-1] if source.archive_members else source.root_file.name
    return PurePosixPath(target).name


def is_village_source(source_name: str, leaf_key: str) -> bool:
    low = source_name.lower().replace("\\", "/")
    return any(
        marker in low
        for marker in (
            "village_data",
            "data_village",
            "/v cses",
            "/vcses",
            "cses2019_village",
        )
    ) or bool(re.search(r"(?:^|/)20\d\dvl_", low)) or leaf_key.startswith("2011vl")


def modules_for_source(source: DataSource, root: Path) -> list[str]:
    name = source.display_name(root)
    low = name.lower().replace("\\", "/")
    leaf = source_leaf(source)
    key = token(PurePosixPath(leaf).stem)
    if "/code/" in low or "allvar" in key or is_village_source(name, key):
        return []

    modules: list[str] = []
    if (
        key in {"areainfo", "areainformation", "households"}
        or "psulisting" in key
        or key.endswith("headinghouseholds")
        or key.endswith("hhotherinfo")
        or key.endswith("hhotherinformation")
    ):
        modules.append("household_core")
    if "hhmembers" in key or key.endswith("s01ahhmember") or key == "members":
        modules.append("household_members")
    if any(marker in key for marker in ("weighthouseholds", "weighthousehold", "sizehouseholds")):
        modules.append("household_weights")
    if any(marker in key for marker in ("weightpersons", "weightindividual", "weighthhpersons")):
        modules.append("person_weights")
    if "personeducation" in key or key.endswith("hhs02education") or key in {"02education", "education"}:
        modules.append("education")
    if any(marker in key for marker in ("personecocurrent", "ecocurrent", "labor7days")):
        modules.append("employment_current")
    if any(marker in key for marker in ("personecousual", "ecousual", "labor12months")):
        modules.append("employment_usual")
    if "migration" in key:
        modules.append("migration")
    if "foodconsumption" in key:
        modules.append("food_consumption")
    if "recallnonfood" in key or "nonfoodexpenses" in key:
        modules.append("nonfood_consumption")
    if "otherfoodsecurity" in key:
        modules.append("food_security")
    if "vulnerability" in key:
        modules.append("vulnerability")
    if key.endswith("hhhousing") or key in {"housing", "04hhhousing"}:
        modules.append("housing")
    if "landownership" in key or key.endswith("landown"):
        modules.append("agriculture_land")
    if any(marker in key for marker in ("productioncrops", "productcrop", "cropsproduction")):
        modules.append("agriculture_crop_production")
    if any(marker in key for marker in ("costcultivation", "costcrops")):
        modules.append("agriculture_crop_costs")
    if "salescrops" in key or "cropsales" in key:
        modules.append("agriculture_crop_sales")
    return sorted(set(modules))


def read_sample(source: DataSource) -> tuple[pd.DataFrame, dict[str, str], int]:
    input_obj: Path | io.BytesIO = (
        io.BytesIO(source.read_bytes()) if source.archive_members else source.root_file
    )
    reader = pd.io.stata.StataReader(input_obj, convert_categoricals=False)
    labels = reader.variable_labels()
    nobs = int(getattr(reader, "nobs", 0) or 0)
    frame = reader.read(nrows=SAMPLE_ROWS)
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    return frame, labels, nobs


def compact_sample(values: Iterable[object]) -> str:
    cleaned = []
    for value in values:
        text = str(value).replace("\n", " ").replace("\r", " ").replace("|", "/")
        cleaned.append(text[:30])
    return ", ".join(cleaned)


def markdown_safe(value: object, limit: int = 80) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "/")[:limit]


def write_readme(
    path: Path,
    variables: pd.DataFrame,
    manifest: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    lines = [
        "# CSES Data Preprocessing - Variable List",
        "",
        "The CSES archives are scanned in memory; `data/raw/` is not modified.",
        "Feasibility is `unknown` because no feasibility-check inventory exists yet.",
        "",
        f"Core module sources scanned: {len(manifest)}",
        f"Variable rows: {len(variables)}",
        f"Read errors: {len(errors)}",
        "",
        "## Module Summary",
        "",
        "| module | title | waves | source datasets | variable rows |",
        "|---|---|---:|---:|---:|",
    ]
    for module, group in manifest.groupby("module", sort=True):
        lines.append(
            f"| {module} | {MODULE_TITLES[module]} | {group['survey_wave'].nunique()} "
            f"| {len(group)} | {len(variables.loc[variables['module'].eq(module)])} |"
        )

    lines.extend(["", "## Datasets and Variables", ""])
    for (module, source_dataset), group in variables.groupby(
        ["module", "source_dataset"], sort=True
    ):
        wave = group["survey_wave"].iloc[0]
        lines.extend(
            [
                f"### `{module}` - `{wave}` - `{source_dataset}`",
                "",
                "| original_name | label | dtype | null_pct | feasibility_status | sample_values |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for row in group.itertuples(index=False):
            lines.append(
                f"| {markdown_safe(row.original_name)} | {markdown_safe(row.variable_label)} "
                f"| {row.dtype} | {row.null_pct}% | {row.feasibility_status} "
                f"| {markdown_safe(row.sample_values)} |"
            )
        lines.append("")

    if not errors.empty:
        lines.extend(["## Read Errors", ""])
        for row in errors.itertuples(index=False):
            lines.append(f"- `{row.source_dataset}`: {markdown_safe(row.error, 240)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)

    variable_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, str]] = []
    module_counts: Counter[str] = Counter()

    for source in discover_sources(root):
        modules = modules_for_source(source, root)
        if not modules:
            continue
        source_name = source.display_name(root)
        wave = normalize_wave(source_name)
        try:
            sample, labels, nobs = read_sample(source)
        except Exception as exc:
            error_rows.append({"source_dataset": source_name, "error": str(exc)})
            continue

        for module in modules:
            manifest_rows.append(
                {
                    "module": module,
                    "module_title": MODULE_TITLES[module],
                    "survey_wave": wave,
                    "source_dataset": source_name,
                    "row_count": nobs,
                    "variable_count": len(sample.columns),
                }
            )
            module_counts[module] += 1
            for column in sample.columns:
                series = sample[column]
                null_pct = round(float(series.isna().mean() * 100), 1) if len(series) else 0.0
                variable_rows.append(
                    {
                        "source_dataset": source_name,
                        "original_name": str(column),
                        "dtype": str(series.dtype),
                        "non_null_count": int(series.notna().sum()),
                        "null_pct": null_pct,
                        "sample_values": compact_sample(series.dropna().head(3)),
                        "feasibility_status": "unknown",
                        "readable_name": "",
                        "full_name": "",
                        "is_final_variable": "",
                        "module": module,
                        "module_title": MODULE_TITLES[module],
                        "survey_wave": wave,
                        "source_rows": nobs,
                        "variable_label": str(labels.get(column, "") or ""),
                    }
                )

    variables = pd.DataFrame(variable_rows)
    manifest = pd.DataFrame(manifest_rows)
    errors = pd.DataFrame(error_rows, columns=["source_dataset", "error"])
    if variables.empty:
        raise RuntimeError("No core CSES variables could be read from the archives")

    variables = variables.sort_values(
        ["module", "survey_wave", "source_dataset", "original_name"]
    )
    manifest = manifest.sort_values(["module", "survey_wave", "source_dataset"])
    variables.to_csv(output / "variable_list.csv", index=False)
    manifest.to_csv(output / "cses_core_module_manifest.csv", index=False)
    errors.to_csv(output / "cses_archive_read_errors.csv", index=False)
    write_readme(output / "README.md", variables, manifest, errors)

    if args.pretty:
        print(f"sources_scanned={len(manifest)}")
        print(f"variable_rows={len(variables)}")
        print(f"read_errors={len(errors)}")
        for module in MODULE_TITLES:
            if module_counts[module]:
                waves = manifest.loc[manifest["module"].eq(module), "survey_wave"].nunique()
                rows = len(variables.loc[variables["module"].eq(module)])
                print(
                    f"{module}: waves={waves}, sources={module_counts[module]}, "
                    f"variable_rows={rows}"
                )
        print("variable_list=data/exp/database/variable_list.csv")
        print("readme=data/exp/database/README.md")


if __name__ == "__main__":
    main()
