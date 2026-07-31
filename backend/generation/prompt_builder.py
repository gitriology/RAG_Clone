"""
Phase 10.3
Prompt Builder

Constructs a grounded prompt for the LLM.

Input
-----
Query
Evidence paragraph

Output
------
Prompt string

The PromptBuilder never calls the LLM.
Its only responsibility is prompt construction.
"""

from typing import Dict


class PromptBuilder:
    """
    Builds a grounded prompt for answer generation.
    """

    def __init__(self):

        self.system_prompt = """
You are a factual question-answering assistant.

Rules:

1. Answer ONLY using the supplied evidence.

2. Do NOT invent information.

3. If the evidence is insufficient, explicitly say:
   "The available evidence is insufficient to answer the question."

4. Keep answers concise.

5. Use complete natural sentences.

6. Do not mention retrieval, documents,
   context or evidence in your response.

7. Preserve factual accuracy.
""".strip()

    # =====================================================
    # PUBLIC
    # =====================================================

    def build(
        self,
        query: str,
        evidence: Dict,
    ) -> str:
        """
        Builds the complete prompt.

        Parameters
        ----------
        query

        evidence

            Output of Phase 10.2
        """

        paragraph = evidence.get(
            "paragraph",
            ""
        ).strip()

        sentence_count = evidence.get(
            "sentence_count",
            0,
        )

        prompt = f"""{self.system_prompt}

Question:
{query}

Evidence:
{paragraph}

Evidence Sentences:
{sentence_count}

Task:

Write a concise answer
using only the evidence above.

Answer:
"""

        return prompt


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    builder = PromptBuilder()

    evidence = {

        "paragraph":
        (
            "The World Health Organization "
            "is a specialized agency of the "
            "United Nations responsible for "
            "international public health."
        ),

        "sentence_count": 2

    }

    prompt = builder.build(

        query="What is WHO?",

        evidence=evidence,

    )

    print()

    print("=" * 70)

    print(prompt)

    print("=" * 70)