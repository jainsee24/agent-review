# AgentReview - Multi-Agent Academic Paper Review

This project reviews academic papers using specialized Claude subagents coordinated
by the Claude Agent SDK. It produces OpenReview-style reviews with ratings.

## Architecture

The orchestrator agent coordinates 11 specialized subagents:

1. **parser** - Extracts PDF to markdown using mineru, processes supplementary files
2. **injection_detector** - Scans for hidden text, prompt injections, steganographic characters
3. **novelty** - Deep web search for related work, assesses contribution novelty
4. **baselines** - Finds missing baselines from recent top-venue papers
5. **code_reviewer** - Reviews provided code for paper-code consistency (optional)
6. **soundness** - Evaluates math, methodology, statistical rigor
7. **writing_quality** - Evaluates grammar, clarity, structure, figures/tables, academic style
8. **reproducibility** - Checks datasets, hyperparameters, compute, seeds, evaluation protocols
9. **ethics_limitations** - Checks ethics statements, broader impact, limitations, dual-use concerns
10. **synthesizer** - Combines all reports into final OpenReview-style review
11. **review_checker** - Validates final review for accuracy, rating consistency, and style compliance

## Pipeline Flow

```
Phase 1: parser (sequential)
Phase 2: [injection_detector, novelty, baselines, soundness, writing_quality, reproducibility, ethics_limitations, code_reviewer] (PARALLEL)
Phase 3: synthesizer (sequential)
Phase 4: review_checker (sequential)
```

Analysis agents in Phase 2 run in parallel for speed.

## Key Directories

- `workspace/` - Per-run working files (parsed paper, reports, final review)
- `agents/` - Agent definitions and system prompts
- `templates/` - Web UI HTML

## Output Files (per run in workspace/)

- `parsed_paper.md` - Full extracted paper text
- `paper_metadata.json` - Title, authors, sections
- `injection_report.json` - Hidden text findings
- `novelty_report.json` - Contribution novelty analysis
- `baselines_report.json` - Missing baselines
- `code_review_report.json` - Code issues (if code provided)
- `soundness_report.json` - Math/methodology assessment
- `writing_quality_report.json` - Writing quality assessment
- `reproducibility_report.json` - Reproducibility evaluation
- `ethics_report.json` - Ethics & limitations assessment
- `final_review.md` - Complete OpenReview-style review
- `final_review.json` - Structured ratings
- `review_checker_report.json` - Quality check results and corrections

## Usage

```bash
# CLI
python main.py --pdf /path/to/paper.pdf
python main.py --pdf /path/to/paper.pdf --model opus --code /path/to/code/

# Web UI
python web.py  # http://localhost:5008
```
