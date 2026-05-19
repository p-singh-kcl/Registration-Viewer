# Contributing

Thanks for thinking about contributing! Issues, bug reports, suggestions, and pull requests are all welcome.

## Reporting bugs

Open a GitHub issue and include, where possible:

- Operating system and Python version (`python --version`).
- Versions of the listed packages (`pip freeze | grep -Ei "nibabel|numpy|matplotlib|scipy|scikit-image"`).
- A minimal description of what you ran and what you expected.
- The full traceback if you got one.
- The shape / orientation of the input volumes (`nib.load(path).shape`, `nib.aff2axcodes(img.affine)`).

If the bug is data-dependent, a small **anonymised** NIfTI that reproduces it is the fastest way to a fix. Please do **not** attach identifiable patient data.

## Suggesting features

Open an issue describing:

- The use case (what are you trying to look at / measure?).
- Why the current viewer doesn't already do it.
- Roughly what the new behaviour should look like.

## Pull requests

1. Fork the repo and create a topic branch: `git checkout -b feature/short-name`.
2. Keep the viewer a **single file** unless there's a strong reason to split it — easy installation is one of the project's goals.
3. Match the existing style: 4-space indent, descriptive function names, docstrings on public helpers, no new top-level dependencies unless they're well-justified.
4. Test the change manually with at least one pair of real NIfTI volumes (with and without labels / predictions where relevant). The viewer is GUI-heavy, so there's no automated test suite — visual verification is expected.
5. Update `README.md` if you change behaviour, add a new window, or add new CLI flags.
6. Open the PR with a short description of what changed and a screenshot if the UI is affected.

## Scope

This tool is meant to stay **focused on visual + quantitative comparison of two co-registered 3D volumes**. Suggestions that broaden that scope (4D time-series, surface rendering, DICOM I/O, training/inference pipelines) are better suited to a separate project — but please open an issue to discuss before sending a large PR.

## Code of conduct

Be kind, be precise, assume good faith. That's it.
