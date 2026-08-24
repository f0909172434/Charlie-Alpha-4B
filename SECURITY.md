# Security policy

Do not run generated code outside an isolated environment. The included evaluator targets macOS
`sandbox-exec` and is a defense-in-depth tool, not a general-purpose hostile-code containment
guarantee.

The v0.3 statistics agent does not execute arbitrary model-written programs. It maps a validated
analysis plan to a checked-in Python or R implementation and copies up to three user-selected data
files into a new temporary directory. Each analysis allows at most four calls. Every call has a
20-second wall limit, 2 GiB memory limit, 32 MiB write limit, and 64 KiB combined output limit;
networking, reads from other user directories, writes outside the temporary directory, and
unapproved executables are denied. The local API binds to `127.0.0.1` unless the operator explicitly
changes it.

Supported inputs are CSV, TSV, JSON, and Parquet, with a 25 MiB per-file and 50 MiB aggregate cap.
Data is processed locally and is not sent to the 9B wording model or any external service. macOS
sandboxing does not replace review by a qualified statistician, particularly for medical,
financial, or policy decisions.

Please report vulnerabilities through GitHub private vulnerability reporting when available. Do
not include credentials, private data, or unredacted machine identifiers in an issue.
