Push local thesis changes to Overleaf and update the parent repo pointer.

Follow this sequence:

**Step 1 — Suggest a commit message:**
Based on the conversation history, propose a concise commit message (imperative mood, ≤72 chars, e.g. "add related work section on trajectory segmentation"). Ask the user to confirm or amend it before proceeding.

**Step 2 — Commit and push inside the submodule:**
```bash
git -C thesis add .
git -C thesis commit -m "<confirmed message>"
git -C thesis push
```

**Step 3 — Update the submodule pointer in the parent repo:**
```bash
git add thesis
git commit -m "update thesis submodule pointer"
```

If there are no uncommitted changes in the thesis submodule, inform the user and stop — do not create empty commits.
