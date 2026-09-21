"""Structured Output Evaluator Example.

Field-by-field scoring of structured output against ground truth.
Deterministic and offline: no LLM judge, no credentials, no per-call cost.

Requires the optional extra:

    pip install "strands-agents-evals[stickler]"
"""

import asyncio

from pydantic import BaseModel
from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import StructuredOutput

# --- The model the agent already emits as structured_output_model ---


class LineItem(BaseModel):
    sku: str | None = None
    quantity: int | None = None


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    total_amount: float | None = None
    line_items: list[LineItem] = []


DOCUMENT = """
INVOICE  #INV-2024-0042
Acme Corporation      Date: 2024-03-15
2x Wireless Mouse (WM-100) @ $29.99
5x USB-C Cable 1m (UC-050) @ $12.99
Total: $124.93
"""


def extract(case: Case) -> Invoice:
    agent = Agent(callback_handler=None)
    result = agent(case.input, structured_output_model=Invoice)
    return result.structured_output


# --- Ground truth: the labelled expected output for this document ---

cases = [
    Case(
        name="invoice-0042",
        input=f"Extract the invoice:\n{DOCUMENT}",
        expected_output=Invoice(
            invoice_id="INV-2024-0042",
            vendor_name="Acme Corporation",
            total_amount=124.93,
            line_items=[
                LineItem(sku="WM-100", quantity=2),
                LineItem(sku="UC-050", quantity=5),
            ],
        ),
    ),
]

# Pass the model for a single-schema suite; omit it to infer per case.
evaluator = StructuredOutput(Invoice)

experiment = Experiment(cases=cases, evaluators=[evaluator])


async def main():
    report = await experiment.run_evaluations_async(extract)
    report.run_display()

    # EvaluationOutput holds four scalars; the detail lives on the evaluator.
    for row in evaluator.per_case():
        print(row["case"], row["overall_score"], row["recall"], row["field_scores"])

    # Per-model-class rollup; per-field counts live under .field_metrics.
    print(evaluator.metrics())

    # Which comparator was chosen for each field, and why.
    print(evaluator.explain())


if __name__ == "__main__":
    asyncio.run(main())
