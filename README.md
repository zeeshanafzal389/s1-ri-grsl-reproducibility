# Sentinel-1 Urban Change Reproducibility Package

This repository provides the code, processed data, and figures supporting the associated IEEE Geoscience and Remote Sensing Letters (GRSL) manuscript:

**Observation Density Governs Sentinel-1 Urban-Change Signal Reproducibility Across Global South Cities**

The manuscript file is not included in this repository. A publication link or DOI can be added after acceptance/publication.

## Contents

- `code/`
  - Python code used to build the Sentinel-1 city-month coverage table, compute S1-RI, run sensitivity and validation analyses, run the downsampling experiment, and generate figures.
- `data/city_counts/`
  - Processed city-month Sentinel-1 coverage table used by the index.
- `data/ri_outputs/`
  - Per-city S1-RI table, regional summaries, sensitivity tables, validation tables, and figure-input tables.
- `data/downsampling/`
  - Per-scene city means, downsampling summaries, and structured-gap robustness outputs.
- `data/external/`
  - Sentinel-1/Open Buildings detectability table used for downstream validation.
- `data/source_city_data/`
  - City attribute table used to join city identifiers, regions, and population/area attributes.
- `figures/`
  - Final 600-dpi PNG and vector PDF figures generated from the processed tables.

## Manuscript Figures

- Figure 1: `figures/figure1_global_s1_ri_map.pdf`
- Figure 2: `figures/figure2_regional_monthly_acquisitions.pdf`
- Figure 3: `figures/figure3_downsampling_dose_response.pdf`
- Figure 4: `figures/figure4_structured_gap_robustness.pdf`

## Key Tables

- Per-city index: `data/ri_outputs/s1_city_reliability_index.csv`
- City-month coverage: `data/city_counts/s1_city_monthly_counts_2017_2026.csv`
- Publication Table I: `data/ri_outputs/table1_reliability_metrics_by_region_and_era_publication.csv`
- Downsampling city-level summary: `data/downsampling/downsample_city_summary_full.csv`
- Dose-response summary: `data/downsampling/dose_response_summary_full.csv`
- Structured-gap robustness: `data/downsampling/structured_gap_dose_response.csv`
- Downstream validation: `data/ri_outputs/s1_ri_detectability_validation_city_table.csv`

## Reproducing the Main Analyses

Use Python 3.10+ from the repository root. Install the dependencies listed in `requirements.txt`.

The processed outputs used by the manuscript are included. To rerun the main analyses:

```bash
python code/build_s1_reliability_index.py --root .
python code/build_s1_sensitivity_analysis.py
python code/run_downsampling_offline_fast.py
python code/build_structured_gap_robustness_figure.py
python code/build_s1_publication_figures.py --root .
```

To rebuild the city-month count table from raw Copernicus metadata, supply the raw parquet inventory and city-boundary GeoJSON locally, then run:

```bash
python code/build_s1_city_monthly_counts.py --root . --city-geojson <path-to-city-boundaries.geojson>
```

For the optional pass-level recount:

```bash
python code/build_s1_city_monthly_counts.py --root . --city-geojson <path-to-city-boundaries.geojson> --pass-level
```

The downsampling experiment uses fixed random seed `20260609` in `code/run_downsampling_offline_fast.py`.

## Large Inputs

The Sentinel-1 scene inventory was derived from Copernicus Data Space Sentinel-1 IW GRDH catalogue metadata for 2017-01 through 2026-05. Large raw parquet inventory files and city-boundary GeoJSON files are not included in this repository. The processed city-month table and analysis-ready downstream tables are included.

## Citation

If this repository is used before journal acceptance, cite
[`zeeshanafzal389/s1-UrbanChange`](https://github.com/zeeshanafzal389/s1-UrbanChange)
as the reproducibility package for the associated IEEE GRSL manuscript. A
Zenodo DOI can be minted after acceptance or public release.

## License and Terms

Code is released under the MIT license. Derived data files are provided for scholarly review and reproducibility and remain subject to the terms of the upstream data providers.
