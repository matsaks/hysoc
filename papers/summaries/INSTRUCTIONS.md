# Paper Summary Instructions

## Workflow

1. **Input**: User provides a research paper (link, PDF, or content).
2. **Summary**: Write a markdown summary and save it in `papers/summaries/`.
3. **Bibliography**: Check if the paper exists in `thesis/bib/bibliography.bib`. If not, add it.

## Summary Content

Each summary should capture:

- **Key findings** of the paper
- **Results**, including figures and tables of results (these are important)
- **Methodology** and main contributions
- **Limitations** and future work (if notable)

## Bibliography Style

Follow the existing BibTeX style in `bibliography.bib`:

- **Citation keys**: `<Paper_Topic_or_Title>` (reflecting what the paper is about, rather than the authors)
- **`key` field**: Formatted as `Author \& Author YYYY` or `Author \textit{et al.} YYYY`
- **Capitalization**: Use `{}` braces to protect proper nouns and acronyms (e.g., `{T}witter`, `{IEEE}`)
- **Standard fields**: Include `doi`, `pages`, `month`, `publisher`, `address`, etc. where available
- **Month**: Use three-letter abbreviations without braces (e.g., `Jan`, `Jun`, `Oct`)

## File Naming

- Summary files: `papers/summaries/<Paper_Topic_or_Title>.md` (matching the new citation key)