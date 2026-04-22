Run the following three commands in sequence to pull the latest Overleaf changes into the thesis submodule and update the parent repo pointer:

```bash
git submodule update --remote thesis
git add thesis
git commit -m "update thesis submodule pointer"
```

Execute all three commands using the Bash tool. Report the output of each step. If `git submodule update` reports no changes, skip the add/commit and inform the user there was nothing to update.
