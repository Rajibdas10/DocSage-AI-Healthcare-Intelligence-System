from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
import re

# Set safe character limit for Groq's 6000 token constraint (~4 chars/token)
MAX_CONTEXT_CHARS = 15000


def find_exact_code(context, question):
    # Extract ICD code from the question using regex
    match = re.search(r'\b([A-Z]\d{3,4})\b', question.upper())
    if match:
        code = match.group(1)
        # Search context for pattern like A032 – Shigellosis
        pattern = re.compile(rf'\b{code}\s*[-–]\s*(.+)', re.IGNORECASE)
        result = pattern.search(context)
        if result:
            return f"{code} – {result.group(1).strip()}"
    return None


def query_langchain(context, question):
    # Truncate context safely but keep more content
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[context truncated]"

    exact_match = find_exact_code(context, question)
    if exact_match:
        return f"✅ Found from context:\n{exact_match}"

    llm = ChatGroq(
        temperature=0.1,  # Lower temperature for more precise answers
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt_template = PromptTemplate.from_template("""
You are a medical coding expert analyzing ICD-10 codes and medical documentation.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

Instructions:
- Only give answers if an **exact match** for the code (like "A021") is found in the CONTEXT.
- Avoid guessing or using similar codes.
- If not found, say: "The code is not present in the context."

Answer:
""")

    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})