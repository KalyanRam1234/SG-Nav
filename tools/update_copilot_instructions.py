#!/usr/bin/env python3
"""
Update .github/copilot-instructions.md with fresh context from docs/ and git history.

Usage:
    python tools/update_copilot_instructions.py
    python tools/update_copilot_instructions.py --model llama3.1
    python tools/update_copilot_instructions.py --dry-run   # print prompt, don't update
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()

INSTRUCTIONS_PATH = os.path.join(REPO_ROOT, ".github", "copilot-instructions.md")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
MAX_DOC_CHARS = 3000  # per doc file, to stay within context limits


def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def get_docs_content():
    """Read all markdown files in docs/."""
    sections = []
    if not os.path.isdir(DOCS_DIR):
        return ""
    for name in sorted(os.listdir(DOCS_DIR)):
        if not name.endswith(".md"):
            continue
        content = read_file(os.path.join(DOCS_DIR, name))
        if len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS] + "\n... (truncated)"
        sections.append(f"### {name}\n\n{content}")
    return "\n\n---\n\n".join(sections)


def get_git_history():
    """Get detailed commit log from the repo author."""
    author = subprocess.check_output(
        ["git", "config", "user.name"], text=True
    ).strip()

    # Recent commits with full messages and file lists
    log = subprocess.check_output(
        [
            "git", "--no-pager", "log",
            f"--author={author}",
            "--format=--- %h %ad ---\n%s\n%b",
            "--date=short",
            "--stat",
            "-50",  # last 50 commits
        ],
        text=True,
        cwd=REPO_ROOT,
    )
    # Cap total size
    if len(log) > 12000:
        log = log[:12000] + "\n... (truncated)"
    return log


def build_prompt(current_instructions, docs_content, git_history):
    return f"""You are updating a .github/copilot-instructions.md file for a codebase.

Below is the CURRENT instructions file, followed by documentation files from docs/, and the developer's git commit history.

Your task: Rewrite the instructions file to incorporate important context from the docs and commit history. Specifically:
- Keep the existing architecture/conventions sections but update them if the commits reveal changes.
- Add a "Development History & Decisions" section summarizing key decisions, experiments, and ongoing work visible from the commits and docs.
- Add a "Known Issues & Workarounds" section if the docs/commits reveal any.
- Keep it factual — only include what the evidence supports.
- Output ONLY the new markdown file content. No preamble, no code fences wrapping the whole file.

---

## CURRENT INSTRUCTIONS FILE

{current_instructions}

---

## DOCUMENTATION FILES (docs/)

{docs_content}

---

## GIT COMMIT HISTORY (most recent first)

{git_history}

---

Now output the updated .github/copilot-instructions.md content:"""


def call_ollama(prompt, model):
    """Call Ollama and return the response text."""
    import ollama as ol

    response = ol.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 16384},
    )
    return response["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description="Update copilot-instructions.md with context from docs/ and git history"
    )
    parser.add_argument(
        "--model", default="llama3.2-vision",
        help="Ollama model to use (default: llama3.2-vision)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the prompt instead of calling the LLM"
    )
    args = parser.parse_args()

    print("📖 Reading current instructions...")
    current = read_file(INSTRUCTIONS_PATH)

    print("📂 Reading docs/...")
    docs = get_docs_content()

    print("📜 Reading git history...")
    history = get_git_history()

    prompt = build_prompt(current, docs, history)

    if args.dry_run:
        print("\n=== PROMPT ===\n")
        print(prompt)
        print(f"\n=== Prompt length: {len(prompt)} chars ===")
        return

    print(f"🤖 Calling Ollama ({args.model})... this may take a minute.")
    try:
        result = call_ollama(prompt, args.model)
    except Exception as e:
        print(f"❌ Ollama call failed: {e}", file=sys.stderr)
        print("Is Ollama running? Try: ollama serve", file=sys.stderr)
        sys.exit(1)

    # Strip any leading/trailing code fences the LLM might add
    result = result.strip()
    if result.startswith("```"):
        result = "\n".join(result.split("\n")[1:])
    if result.endswith("```"):
        result = "\n".join(result.split("\n")[:-1])
    result = result.strip()

    os.makedirs(os.path.dirname(INSTRUCTIONS_PATH), exist_ok=True)
    with open(INSTRUCTIONS_PATH, "w") as f:
        f.write(result + "\n")

    print(f"✅ Updated {INSTRUCTIONS_PATH}")
    print(f"   Review with: git diff .github/copilot-instructions.md")


if __name__ == "__main__":
    main()
