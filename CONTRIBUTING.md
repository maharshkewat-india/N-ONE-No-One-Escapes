# Contributing to N-ONE Surveillance Platform

First off, thank you for considering contributing to N-ONE! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, please open a new issue. It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If you're ready to contribute, fork the repository and create a new branch with a descriptive name.

```sh
# A good branch name would be (where issue #123 is the ticket you're working on)
git checkout -b 123-fix-bug-in-retinaface
```

## How to contribute

### Reporting Bugs

This section guides you through submitting a bug report for N-ONE. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

*   **Use a clear and descriptive title** for the issue to identify the problem.
*   **Describe the exact steps which reproduce the problem** in as much detail as possible.
*   **Provide specific examples to demonstrate the steps.** Include copy/pasteable snippets if possible.
*   **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for N-ONE, including completely new features and minor improvements to existing functionality.

*   **Provide a step-by-step description of the suggested enhancement** in as much detail as possible.
*   **Explain why this enhancement would be useful** to most N-ONE users.

### Pull Request Process

1.  Ensure your code adheres to the project's coding standards.
2.  Update the `README.md` or other relevant documentation with details of changes, if any.
3.  Ensure your pull request has a clear title and description of the changes.

Thank you for your contribution!

## Current feature and documentation expectations

When changing recognition or dashboard behavior, update the relevant root-level documentation as well as `app.py`. In particular, keep these rules synchronized:

- Registered profiles use `Staff` and `Victim`; retain compatibility with `Member_` and `Lost_` files unless a migration is explicitly planned.
- Both Administrator and Operator dashboards expose registered/unknown counts and the registered-face inventory filters.
- Victim Search must remain target-specific: only the selected Victim can be displayed as a visible match.
- Unknown faces must continue to be stored/re-identified silently during Victim Search.
- Camera location belongs in unknown sightings and Victim sighting history.
- Document user-visible Streamlit connection caveats such as Windows `WinError 10054` when the browser closes during the live loop.

Before opening a pull request, run:

```powershell
python -m pytest tests -q
python -m py_compile app.py deepface_adapter.py
```
