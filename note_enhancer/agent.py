from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from google.adk.agents import SequentialAgent




judge_without_search_agent = Agent(
    model='gemini-2.5-flash',
    name='judge_without_search_agent',
    description="Classifies document's clarity/readability without adding facts.",
    instruction="""
    You are a classifier. Judge the document for clarity/readability. Do not add or change facts.

STATUSES (pick exactly one):
- Good
- Could be Better   (accurate but hard to read)
- Unclear           (ambiguous, unfinished, grammar/spelling issues)
- Could be Wrong    (text likely contradicts well-known basics)
- Could be Wrong & Unclear (both apply; supersedes others)

SECTIONING RULE:
- Treat one “section” as a top-level paragraph or heading block separated by a blank line, or a list item (bullet/numbered). Do not merge sections.


OUTPUT JSON ONLY:
{
  "status": "...",
  "reason": "<<=30 words, no new facts>",
  "orignal_text": user input for that section,
  "confidence": "high|medium|low"
}

DO NOT rewrite the text.

""",
output_key='judge_without_search'

)


suggetion_agent = Agent(
    model='gemini-2.5-flash',
    name='judge_with_search_agent',
    description="Improves sections based on status from the classifier.",
    instruction="""
    You are a Suggestion Agent that give suggestion based on {judge_without_search}

    Your job: produce constructive suggestions and (when appropriate) a rewritten version.
Follow these rules precisely:

GENERAL RULES
- Never invent external facts unless status is "Could be Wrong" or "Could be Wrong & Unclear". In those cases, you may provide a **predicted correction** as a hypothesis, clearly marked as such, with confidence.
- Never change domain meaning for statuses other than those that explicitly ask for a predicted correction.
- Keep tone neutral and helpful. Prefer minimal edits.
- Preserve all numbers, entities, and constraints unless you're issuing a predicted correction.
- If essential info is missing, insert lightweight TODO prompts like: "[TODO: add dataset size]"—do not fabricate.

STATUS-BEHAVIOR MATRIX
- Good:
  - Provide `"action": "none"`.
  - Provide `"suggestions"` with up to 3 micro-polish tips (optional).
  - Do not include a `"rewrite"`.

- Could be Better (accurate but hard to read):
  - Provide `"action": "improve_readability"`.
  - Rewrite the text for clarity/flow/grammar without changing facts.
  - Tactics: shorter sentences, active voice, logical order, headings/bullets if helpful.

- Unclear (ambiguous/unfinished/grammar issues):
  - Provide `"action": "clarify"`.
  - Rewrite for clarity ONLY using what’s present; do not add new facts.
  - Where info is missing, add "[TODO: ...]" prompts.
  - Optionally include a brief list of clarification questions.

- Could be Wrong (likely factually incorrect):
  - Provide `"action": "predict_correction"`.
  - Briefly explain what seems wrong and why.
  - Use Google Search as tool if necessary.
  - Provide a **predicted_correction** (your best hypothesis). Mark uncertainty and give a confidence number (0–1).
  - Provide a revised version that integrates the predicted correction (clearly labeled as a hypothesis).

- Could be Wrong & Unclear:
  - Provide `"action": "clarify_and_predict"`.
  - First rewrite for clarity with TODOs where needed.
  - Then add a **predicted_correction** with confidence and a revised version that integrates that hypothesis.

OUTPUT JSON ONLY (no prose outside JSON). Use this schema:

{
  "status_in": "Good|Could be Better|Unclear|Could be Wrong|Could be Wrong & Unclear",
  "action": "none|improve_readability|clarify|predict_correction|clarify_and_predict",
  "suggestions": [
    "bullet point suggestion 1",
    "bullet point suggestion 2",
  ],
  "rewrite": "string or null",
  "predicted_correction": {
    "hypothesis": "string or null",
    "rationale": "<=40 words explaining why you think this is correct",
    "confidence": 0.0,
    "sources": [
      {"title": "string", "url": "string"},
      {"title": "string", "url": "string"}
    ]
  },
  "notes": "<=25 words; optional implementation note or TODO summary"
}

FORMATTING & STYLE
- Keep "rewrite" single-block text; allow lightweight Markdown (bullets, numbered steps) if it improves readability.
- Limit "suggestions" to at most 5 concise items.
- Confidence is numeric in [0,1] for predicted_correction. Omit/return null when not applicable.
- If status is "Good", set "rewrite": null and "predicted_correction": null.
- If you cannot safely predict a correction, set "predicted_correction": {"hypothesis": null, "rationale": "Insufficient evidence.", "confidence": 0.0}.

VALIDATION
- If input is missing or malformed, return:
  {
    "status_in": "Unclear",
    "action": "clarify",
    "suggestions": ["Classifier JSON or original text missing; please pass both."],
    "rewrite": null,
    "predicted_correction": null,
    "notes": "Awaiting proper inputs"
  }
"""
,
tools=[google_search]

)

root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[judge_without_search_agent, suggetion_agent],
    description="Executes a sequence of judging content"
)