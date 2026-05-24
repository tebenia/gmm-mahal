# Severi EMBER v1 to EMBER2024 v3 Feature Policy Mapping

This note documents how the original Severi-style 17-feature PE trigger set is
mapped into the EMBER2024 feature layout. The mapping is a research heuristic for
feature-space experiments. It is not proof that an EMBER2024 trigger can be
edited into real PE files without changing functionality.

## Target Groups

| Target group | EMBER2018 v2 count | EMBER2024 v3 count | Meaning |
|---|---:|---:|---|
| `feature_space_feasible` / `feasible` | 47 | 288 | Broad feature-vector candidate set: non-hashed features minus configured exclusions. |
| `problem_space_conservative` | 17 | 19 | Conservative semantic mapping of the Severi 17 into EMBER2024. Includes renamed string-count equivalents. |
| `severi_exact_overlap` | 17 | 12 | Exact-name overlap only. No renamed or semantic substitutions for EMBER2024. |

## Severi 17 Mapping

| Severi v1 feature | EMBER2024 v3 equivalent | Conservative | Exact overlap | Reason |
|---|---|---|---|---|
| `paths_count` | `string_count_38_file_path` | keep | exclude | EMBER2024 replaces broad path counting with regex-specific string counts. This is a close semantic replacement, but not an exact name match. |
| `urls_count` | `string_count_73_url`, `string_count_44_http`, `string_count_45_http`, `string_count_46_https` | keep | exclude | EMBER2024 splits URL-like evidence into several regex counters. These are conservative semantic replacements for URL strings. |
| `registry_count` | `string_count_63_registry_key` | keep | exclude | EMBER2024 uses a registry-key regex counter instead of the old broad registry count. |
| `MZ_count` | none selected | exclude | exclude | EMBER2024 has related fields such as `dos_header_e_magic`, `start_byte0`, `start_byte1`, and `string_count_30_dos_msg`, but none is clearly the same feature. |
| `size` | `size` | keep | keep | Exact name and same broad file-size concept. |
| `timestamp` | `timestamp` | keep | keep | Exact name and same compilation/header timestamp concept. |
| `major_image_version` | `major_image_version` | keep | keep | Exact name and same optional-header version concept. |
| `minor_image_version` | `minor_image_version` | keep | keep | Exact name and same optional-header version concept. |
| `major_linker_version` | `major_linker_version` | keep | keep | Exact name and same optional-header linker version concept. |
| `minor_linker_version` | `minor_linker_version` | keep | keep | Exact name and same optional-header linker version concept. |
| `major_operating_system_version` | `major_operating_system_version` | keep | keep | Exact name and same optional-header OS version concept. |
| `minor_operating_system_version` | `minor_operating_system_version` | keep | keep | Exact name and same optional-header OS version concept. |
| `minor_subsystem_version` | `minor_subsystem_version` | keep | keep | Exact name and same optional-header subsystem version concept. |
| `num_zero_size_sections` | `num_zero_size_sections` | keep | keep | Exact name and same section-count concept. |
| `num_unnamed_sections` | `num_empty_name_sections` | keep | exclude | EMBER2024 renames the concept from unnamed sections to empty-name sections. |
| `num_read_and_execute_sections` | `num_read_and_execute_sections` | keep | keep | Exact name and same section-permission count concept. |
| `num_write_sections` | `num_write_sections` | keep | keep | Exact name and same section-permission count concept. |

## Excluded EMBER2024 Families

These are excluded from `problem_space_conservative` even when they appear in
`feature_space_feasible`:

- hashed imports, exports, sections, Rich-header buckets, histograms, byte-entropy bins, and `printabledist`
- Authenticode fields
- PE parser warning fields
- checksum
- data-directory sizes and virtual addresses
- overlay size, overlay ratios, and overlay entropy
- DLL and image characteristic flags
- DOS-header fields
- most COFF/header address and size fields

The strict `severi_exact_overlap` group is useful as an ablation because it asks
how much attack strength remains if EMBER2024 is limited only to feature names
that exactly survived from the original Severi EMBER feature layout.

