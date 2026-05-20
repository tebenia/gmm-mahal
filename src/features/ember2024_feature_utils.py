"""
Feature-name helpers for EMBER2024 / EMBER feature version 3 ("thrember").

This follows the feature layout implemented by EMBER2024's
``thrember.features.PEFeatureExtractor``.
The EMBER2024 paper/README describes feature version 3 as a pefile-based
reimplementation of EMBER with additional DOS header, Rich header, PE data
directory, Authenticode, and PE parsing-warning features.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


NUM_EMBER2024_FEATURES = 2568
NUM_EMBER_FEATURES = NUM_EMBER2024_FEATURES

STRING_REGEX_NAMES = [
    ".click(",
    "/EmbeddedFile",
    "/FlateDecode",
    "/URI",
    "/bin/",
    "/dev/",
    "/proc/",
    "/tmp/",
    "/usr/",
    "<script",
    "Invoke-Command",
    "Invoke-Expression",
    "Start-process",
    "base64",
    "base64string",
    "btc_wallet",
    "cache",
    "certificate",
    "clipboard",
    "command",
    "connect",
    "cookie",
    "create",
    "crypt",
    "debug",
    "decode",
    "delete",
    "desktop",
    "directory",
    "disk",
    "dos_msg",
    "download",
    "email_addr",
    "encode",
    "enum",
    "environment",
    "exit",
    "file",
    "file_path",
    "ftp",
    "get",
    "hidden",
    "hostname",
    "html",
    "http",
    "http://",
    "https://",
    "install",
    "internet",
    "ipv4_addr",
    "ipv6_addr",
    "javascript",
    "keyboard",
    "mac_addr",
    "memory",
    "module",
    "mutex",
    "onlick",
    "password",
    "post",
    "powershell",
    "privilege",
    "process",
    "registry_key",
    "remote",
    "resource",
    "security",
    "service",
    "shell",
    "snapshot",
    "system",
    "thread",
    "token",
    "url",
    "useragent",
    "wallet",
    "window",
]

IMAGE_CHARACTERISTICS = [
    "RELOCS_STRIPPED",
    "EXECUTABLE_IMAGE",
    "LINE_NUMS_STRIPPED",
    "LOCAL_SYMS_STRIPPED",
    "AGGRESIVE_WS_TRIM",
    "LARGE_ADDRESS_AWARE",
    "16BIT_MACHINE",
    "BYTES_REVERSED_LO",
    "32BIT_MACHINE",
    "DEBUG_STRIPPED",
    "REMOVABLE_RUN_FROM_SWAP",
    "NET_RUN_FROM_SWAP",
    "SYSTEM",
    "DLL",
    "UP_SYSTEM_ONLY",
    "BYTES_REVERSED_HI",
]

DLL_CHARACTERISTICS = [
    "HIGH_ENTROPY_VA",
    "DYNAMIC_BASE",
    "FORCE_INTEGRITY",
    "NX_COMPAT",
    "NO_ISOLATION",
    "NO_SEH",
    "NO_BIND",
    "APPCONTAINER",
    "WDM_DRIVER",
    "GUARD_CF",
    "TERMINAL_SERVER_AWARE",
]

DOS_MEMBERS = [
    "e_magic",
    "e_cblp",
    "e_cp",
    "e_crlc",
    "e_cparhdr",
    "e_minalloc",
    "e_maxalloc",
    "e_ss",
    "e_sp",
    "e_csum",
    "e_ip",
    "e_cs",
    "e_lfarlc",
    "e_ovno",
    "e_oemid",
    "e_oeminfo",
    "e_lfanew",
]

DATA_DIRECTORY_NAMES = [
    "EXPORT",
    "IMPORT",
    "RESOURCE",
    "EXCEPTION",
    "SECURITY",
    "BASERELOC",
    "DEBUG",
    "COPYRIGHT",
    "GLOBALPTR",
    "TLS",
    "LOAD_CONFIG",
    "BOUND_IMPORT",
    "IAT",
    "DELAY_IMPORT",
    "COM_DESCRIPTOR",
    "RESERVED",
]

AUTHENTICODE_FEATURE_NAMES = [
    "authenticode_num_certs",
    "authenticode_self_signed",
    "authenticode_empty_program_name",
    "authenticode_no_countersigner",
    "authenticode_parse_error",
    "authenticode_chain_max_depth",
    "authenticode_latest_signing_time",
    "authenticode_signing_time_diff",
]

PE_WARNING_PATTERNS = [
    "AddressOfEntryPoint lies outside the sections' boundaries...",
    "Bad RVA in relocation data...",
    "Byte 0x...",
    "Corrupt header...",
    "Damaged Import Table information...",
    "Don't know how to parse LOAD_CONFIG information for non-PE32...",
    "Error, too many imported symbols...",
    "Error parsing a resource directory data entry...",
    "Error parsing export directory at RVA...",
    "Error parsing resource of type RT_STRING at...",
    "Error parsing StringFileInfo/VarFileInfo struct...",
    "Error parsing the Delay import directory...",
    "Error parsing the Delay import directory at RVA...",
    "Error parsing the import directory at RVA...",
    "Error parsing the import directory. Invalid Import data at RVA...",
    "Error parsing the import table. Entries go beyond bounds...",
    "Error parsing the import table. AddressOfData overlaps with THUNK_DATA for THUNK at RVA...",
    "Error parsing the import table. Invalid data at RVA...",
    "Error parsing the resources directory. Excessively nested table depth...",
    "Error parsing the resources directory. The directory contains...",
    "Error parsing the resources directory. The file contains at least...",
    "Error parsing the resources directory. Entry...",
    "Error parsing the resources directory, attempting to read entry name. Entry names overlap...",
    "Error parsing the resources directory, attempting to read entry name. Can't read unicode string at offset...",
    "Error parsing the version information, attempting to read OffsetToData with RVA...",
    "Error parsing the version information, attempting to read VS_VERSION_INFO string...",
    "Error parsing the version information, attempting to read VarFileInfo Var string...",
    "Error parsing the version information, attempting to read StringFileInfo string...",
    "Error parsing the version information, attempting to read StringTable string...",
    "Error parsing the version information, attempting to read StringTable Key string...",
    "Error parsing the version information, to read StringTable Value string...",
    "Excessive number of imports...",
    "Export directory contains more than 10 repeated entries...",
    "Failed parsing FunctionEntry of UNWIND_INFO at...",
    "Failed rendering pascal string, attempting to read from RVA 0x...",
    "Failed rendering unicode string, attempting to read from RVA 0x...",
    "Failed to process directory...",
    "FunctionEntry of UNWIND_INFO at...",
    "If SectionAlignment...",
    "If FileAlignment > 0x200 it should be a power of 2. Value...",
    "Imported symbols contain entries typical of packed executables...",
    "Invalid bdd dynamic relocation...",
    "Invalid bdd info...",
    "Invalid debug information...",
    "Invalid function override header...",
    "Invalid function override info...",
    "Invalid IMAGE_DYNAMIC_RELOCATION_TABLE information...",
    "Invalid LOAD_CONFIG information...",
    "Invalid relocation information. Can't read...",
    "Invalid relocation information. SizeOfBlock too large...",
    "Invalid relocation information. VirtualAddress outside...",
    "Invalid resources directory. Can't read...",
    "Invalid resources directory. Can't parse directory data at RVA...",
    "Invalid TLS information. Can't read...",
    "Invalid type 0x...",
    "Invalid VS_VERSION_INFO block...",
    "No parsing available for IMAGE_DYNAMIC_RELOCATION_TABLE...",
    "Overlapping offsets in relocation data...",
    "Possibly corrupt file. AddressOfEntryPoint lies outside the file...",
    "Relocating image but PE does not have (or pefile cannot parse) a DIRECTORY_ENTRY_BASERELOC...",
    "Resource size...",
    "Rich Header is malformed...",
    "Rich Header is not in Microsoft format, possibly malformed...",
    "RVA AddressOfFunctions in the export directory points to an invalid...",
    "RVA AddressOfNames in the export directory points to an invalid...",
    "RVA of IMAGE_BOUND_IMPORT_DESCRIPTOR points...",
    "SizeOfHeaders is smaller than AddressOfEntryPoint...",
    "Suspicious flags set for section...",
    "Suspicious NumberOfRvaAndSizes in the Optional Header...",
    "Suspicious value found parsing section...",
    "The Bound Imports directory exists but can't be parsed...",
    "Too many warnings parsing section. Aborting...",
    "Too many errors parsing the Delay import directory...",
    "Too many errors parsing the import directory...",
    "Too many sections...",
    "Unknown UNWIND_CODE at...",
    "Unsupported version of UNWIND_INFO...",
    "...Contents are null-bytes.",
    "...No data in the file (is this corkami's virtsectblXP?).",
    "...PointerToRawData points beyond the end of the file.",
    "...PointerToRawData should normally be a multiple of FileAlignment, this might imply the file is trying to confuse tools which parse this incorrectly.",
    "...SizeOfRawData is larger than file.",
    "...VirtualSize is extremely large > 256MiB",
    "...VirtualAddress is beyond 0x10000000",
    "...symbol entries. Assuming corrupt.",
    "...ordinal entries. Assuming corrupt.",
    "...Assuming corrupt.",
]


def _safe_name(value):
    value = value.lower()
    value = value.replace("0x", "hex_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def _append_range(names, base, prefix, count):
    for i in range(count):
        names[base + i] = "{}{}".format(prefix, i)
    return base + count


def build_feature_names():
    names = [""] * NUM_EMBER2024_FEATURES
    base = 0

    # GeneralFileInfo, dim 7
    names[base + 0] = "size"
    names[base + 1] = "entropy"
    names[base + 2] = "is_pe"
    for i in range(4):
        names[base + 3 + i] = "start_byte{}".format(i)
    base += 7

    # ByteHistogram, dim 256
    base = _append_range(names, base, "ByteHistogram", 256)

    # ByteEntropyHistogram, dim 256
    base = _append_range(names, base, "ByteEntropyHistogram", 256)

    # StringExtractor, dim 177
    names[base + 0] = "numstrings"
    names[base + 1] = "avlength"
    names[base + 2] = "printables"
    for i in range(96):
        names[base + 3 + i] = "printabledist{}".format(i)
    names[base + 99] = "string_entropy"
    for i, regex_name in enumerate(STRING_REGEX_NAMES):
        names[base + 100 + i] = "string_count_{:02d}_{}".format(i, _safe_name(regex_name))
    base += 177

    # HeaderFileInfo, dim 74
    header_names = [
        "timestamp",
        "coff_number_of_sections",
        "coff_number_of_symbols",
        "coff_sizeof_optional_header",
        "coff_pointer_to_symbol_table",
        "machine",
        "subsystem",
        "major_image_version",
        "minor_image_version",
        "major_linker_version",
        "minor_linker_version",
        "major_operating_system_version",
        "minor_operating_system_version",
        "major_subsystem_version",
        "minor_subsystem_version",
        "sizeof_code",
        "sizeof_headers",
        "sizeof_image",
        "sizeof_initialized_data",
        "sizeof_uninitialized_data",
        "sizeof_stack_reserve",
        "sizeof_stack_commit",
        "sizeof_heap_reserve",
        "sizeof_heap_commit",
        "address_of_entrypoint",
        "base_of_code",
        "image_base",
        "section_alignment",
        "checksum",
        "number_of_rvas_and_sizes",
    ]
    for name in header_names:
        names[base] = name
        base += 1
    for characteristic in IMAGE_CHARACTERISTICS:
        names[base] = "image_characteristic_{}".format(_safe_name(characteristic))
        base += 1
    for characteristic in DLL_CHARACTERISTICS:
        names[base] = "dll_characteristic_{}".format(_safe_name(characteristic))
        base += 1
    for member in DOS_MEMBERS:
        names[base] = "dos_header_{}".format(member)
        base += 1

    # SectionInfo, dim 224
    section_general_names = [
        "num_sections",
        "num_zero_size_sections",
        "num_empty_name_sections",
        "num_read_and_execute_sections",
        "num_write_sections",
        "max_section_entropy",
        "min_section_entropy",
        "max_section_size_ratio",
        "min_section_size_ratio",
        "max_section_vsize_ratio",
        "min_section_vsize_ratio",
    ]
    for name in section_general_names:
        names[base] = name
        base += 1
    base = _append_range(names, base, "section_size_hash", 50)
    base = _append_range(names, base, "section_vsize_hash", 50)
    base = _append_range(names, base, "section_entropy_hash", 50)
    base = _append_range(names, base, "section_characteristics_hash", 50)
    base = _append_range(names, base, "section_entry_name_hash", 10)
    names[base + 0] = "overlay_size"
    names[base + 1] = "overlay_size_ratio"
    names[base + 2] = "overlay_entropy"
    base += 3

    # ImportsInfo, dim 1282
    names[base + 0] = "imports"
    names[base + 1] = "import_libraries"
    base += 2
    base = _append_range(names, base, "import_libs_hash", 256)
    base = _append_range(names, base, "import_funcs_hash", 1024)

    # ExportsInfo, dim 129
    names[base] = "exports"
    base += 1
    base = _append_range(names, base, "exports_hash", 128)

    # DataDirectories, dim 34
    for directory_name in DATA_DIRECTORY_NAMES:
        name = _safe_name(directory_name)
        names[base] = "datadirectory_{}_size".format(name)
        names[base + 1] = "datadirectory_{}_virtual_address".format(name)
        base += 2
    names[base] = "has_relocs"
    names[base + 1] = "has_dynamic_relocs"
    base += 2

    # RichHeader, dim 33
    names[base] = "richheader_num_pairs"
    base += 1
    base = _append_range(names, base, "richheader_hash", 32)

    # AuthenticodeSignature, dim 8
    for feature_name in AUTHENTICODE_FEATURE_NAMES:
        names[base] = feature_name
        base += 1

    # PEFormatWarnings, dim 88
    for i, warning in enumerate(PE_WARNING_PATTERNS):
        names[base] = "pe_warning_{:03d}_{}".format(i, _safe_name(warning))
        base += 1
    names[base] = "pe_warning_count"
    base += 1

    assert base == NUM_EMBER2024_FEATURES
    assert all(names)
    assert len(set(names)) == len(names)

    return names


def build_feature_groups():
    """Return feature group index ranges matching thrember's feature order."""

    groups = {}
    base = 0
    for group_name, dim in [
        ("general", 7),
        ("histogram", 256),
        ("byteentropy", 256),
        ("strings", 177),
        ("header", 74),
        ("section", 224),
        ("imports", 1282),
        ("exports", 129),
        ("datadirectories", 34),
        ("richheader", 33),
        ("authenticode", 8),
        ("pefilewarnings", 88),
    ]:
        groups[group_name] = list(range(base, base + dim))
        base += dim
    assert base == NUM_EMBER2024_FEATURES
    return groups


def get_hashed_features():
    feature_names = build_feature_names()
    result = []
    for i, feature_name in enumerate(feature_names):
        if (
            "_hash" in feature_name
            or "Histogram" in feature_name
            or feature_name.startswith("printabledist")
        ):
            result.append(i)
    return result


def get_non_hashed_features():
    hashed = set(get_hashed_features())
    return [i for i in range(NUM_EMBER2024_FEATURES) if i not in hashed]


def get_categorical_features():
    """Return categorical feature indices used by EMBER2024's example trainer."""

    return [2, 3, 4, 5, 6, 701, 702]


def _ensure_thrember_importable():
    try:
        import thrember  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root.parent / "ember2024" / "EMBER2024" / "src"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def get_features_from_file(feature_names, file_path):
    """Return selected EMBER2024 feature values from a local file."""

    if type(feature_names) is str:
        feature_names = [feature_names]

    built_feature_names = build_feature_names()
    feature_ids = [built_feature_names.index(feature_name) for feature_name in feature_names]
    feature_ids = np.array(feature_ids)
    assert all(feature_ids >= 0)

    _ensure_thrember_importable()
    import thrember

    pe_feat_extractor = thrember.PEFeatureExtractor()
    with open(file_path, "rb") as f:
        bytez = f.read()
    feature_values = np.array(pe_feat_extractor.feature_vector(bytez))
    result = feature_values[feature_ids]
    return tuple(result)


if __name__ == "__main__":
    feature_names = build_feature_names()
    print("EMBER2024 feature count:", len(feature_names))
    print("Hashed features:", len(get_hashed_features()))
    print("Non-hashed features:", len(get_non_hashed_features()))
