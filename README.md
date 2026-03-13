# 🔬 AgentReview

**Multi-agent AI framework for academic paper review.** 📝

AgentReview coordinates 11 specialized Claude subagents to produce comprehensive, OpenReview-style peer reviews. Upload a paper PDF (and optionally code as a ZIP), and the system analyzes novelty, soundness, baselines, writing quality, reproducibility, ethics, and more — all in parallel — then synthesizes everything into a structured review with ratings.

Built on the [Claude Agent SDK](https://docs.anthropic.com/en/docs/claude-agent-sdk).

## ✨ Features

- **11 specialized review agents** running in a coordinated pipeline
- **Parallel execution** — up to 8 analysis agents run simultaneously
- **OpenReview-format output** with ratings (soundness, presentation, contribution, originality, confidence) and accept/borderline/reject recommendation
- **Novelty analysis** — deep web search for prior work and contribution classification
- **Baseline search** — finds missing state-of-the-art comparisons from top venues (NeurIPS, ICML, ICLR, CVPR, etc.)
- **Soundness evaluation** — checks mathematical rigor, statistical methodology, and experimental design
- **Writing quality** — evaluates grammar, clarity, structure, figures/tables, citation integrity, reporting standards compliance, and figure integrity
- **Reproducibility check** — verifies datasets, hyperparameters, compute resources, seeds, and evaluation protocols
- **Ethics & limitations** — checks for ethics statements, broader impact, limitations coverage, dual-use concerns, bias assessment, and research integrity
- **Injection detection** — scans for hidden text, white-on-white text, and prompt injections embedded in papers
- **Code review** — optional cross-referencing of provided code against paper claims
- **Quality verification** — final review checker validates factual accuracy and rating consistency
- **Real-time web UI** with live pipeline visualization, agent activity tracking, and SSE streaming
- **CLI interface** for scripted/batch usage
- **Two review modes** — "Peer Review" (reviewing others' work) or "Pre-Submission" (reviewing your own paper before submitting, with actionable revision checklist)
- **Citation integrity** — checks for missing key citations, self-citation bias, and misrepresented references
- **Reporting standards** — checks compliance with CONSORT, STROBE, PRISMA, ARRIVE, and ML reproducibility guidelines
- **API key input** — enter your Anthropic key directly in the web UI (never stored on server)

## 🏗️ Architecture

```
Phase 1: Parser (sequential)          -> PDF to markdown extraction
Phase 2: Analysis (PARALLEL)          -> Up to 8 agents run simultaneously
           |-- Injection Detector
           |-- Novelty Analyzer
           |-- Baseline Searcher
           |-- Soundness Evaluator
           |-- Writing Quality
           |-- Reproducibility Checker
           |-- Ethics & Limitations
           +-- Code Reviewer (optional)
Phase 3: Synthesizer (sequential)     -> Combines all reports into final review
Phase 4: Review Checker (sequential)  -> Quality control and verification
```

## 📋 Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed (provides the Claude Agent SDK)
- [MinerU](https://github.com/opendatalab/MinerU) (recommended for PDF parsing, falls back to `pdftotext`)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/agent-review.git
cd agent-review

# Install dependencies
pip install -r requirements.txt

# (Optional) Install MinerU for better PDF parsing
pip install mineru
```

## 💻 Usage

### Web UI (recommended)

```bash
python web.py
```

Open `http://localhost:5008` in your browser. Enter your Anthropic API key, upload a paper PDF, configure review options, and click **Review Paper**.

### CLI

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Basic review
python main.py --pdf /path/to/paper.pdf

# Use Opus model for highest quality
python main.py --pdf /path/to/paper.pdf --model opus

# Include code review (provide code directory)
python main.py --pdf /path/to/paper.pdf --code /path/to/code/

# Include supplementary files
python main.py --pdf /path/to/paper.pdf --supplementary /path/to/supp.pdf

# Skip specific analyses
python main.py --pdf /path/to/paper.pdf --no-novelty --no-baselines
python main.py --pdf /path/to/paper.pdf --no-writing --no-ethics --no-reproducibility
```

### Models

| Model  | Quality   | Speed  |
|--------|-----------|--------|
| Haiku  | Good      | Fast   |
| Sonnet | Great     | Medium |
| Opus   | Best      | Slow   |

## 📄 Output

Each review run creates a workspace directory containing:

- `final_review.md` — complete OpenReview-style review
- `final_review.json` — structured ratings and scores
- `parsed_paper.md` — extracted paper text
- `novelty_report.json` — contribution analysis
- `baselines_report.json` — missing baseline comparisons
- `soundness_report.json` — methodology evaluation
- `writing_quality_report.json` — writing quality assessment
- `reproducibility_report.json` — reproducibility evaluation
- `ethics_report.json` — ethics & limitations assessment
- `injection_report.json` — hidden text findings
- `review_checker_report.json` — quality verification results

## 📁 Project Structure

```
agent-review/
├── web.py                 # Flask web server with SSE streaming
├── main.py                # CLI entry point
├── config.py              # Pipeline configuration
├── agents/
│   ├── __init__.py
│   └── definitions.py     # All 11 agent definitions & system prompts
├── templates/
│   └── index.html         # Web UI (single-page app)
├── requirements.txt
└── CLAUDE.md
```

## ⚡ How It Compares

| Feature | AgentReview | Stanford Agentic Reviewer | ScholarsReview |
|---------|-------------|--------------------------|----------------|
| Multi-agent parallel analysis | Yes (8 parallel) | No (sequential) | No |
| Writing quality check | Yes | Yes (1 dimension) | Yes |
| Reproducibility check | Yes | No | No |
| Ethics & limitations | Yes | No | No |
| Injection detection | Yes | No | No |
| Code review | Yes | No | No |
| Baseline search (web) | Yes | Yes | No |
| OpenReview format | Yes | Partial | No |
| Real-time streaming UI | Yes | No | No |
| Open source | Yes | Partial | No |

## 📜 License

MIT
