# ContextOps: The Complete User Guide

Welcome to **ContextOps**! If you are building AI applications (like chatbots or RAG systems), this guide will show you exactly how to use ContextOps to make sure you aren't wasting money on bad prompts or confusing your AI.

---

## What is ContextOps?
Think of ContextOps like a spell-checker, but for the information you send to an AI. 

When you send a prompt to an AI, you often attach a lot of background information (like retrieved documents or chat history). If you attach too much junk, duplicate info, or irrelevant data, the AI gets confused and you pay for wasted tokens. ContextOps analyzes your data *before* you send it to the AI and gives you a score from **0 to 100**, along with actionable advice on how to fix it.

---

## Step 1: Installation

You can install ContextOps on your computer just like any other Python tool. Open your terminal (or command prompt) and run:

```bash
pip install contextops
```

To make sure it installed correctly, you can run our built-in demo:
```bash
contextops demo
```
*This will show you a colorful example of a bad prompt getting scored and analyzed!*

---

## Step 2: Prepare Your Data (.json files)

To use ContextOps, you need to save the data you plan to send to the AI in a `.json` file. 

*(Wondering how to actually get this file from your code? Read our [How to Get Your Context JSON Guide](HOW_TO_GET_JSON.md) first!)*

ContextOps understands two formats. Use whichever one is easier for you!

### Format A: The "Structured" Format (Recommended)
This format explicitly tells ContextOps what each piece of text is.

**Example (`my_prompt.json`):**
```json
{
    "system": "You are a helpful customer support bot.",
    "chunks": [
        {"content": "Refunds take 3-5 business days.", "source": "docs/refunds.md"},
        {"content": "We do not offer refunds on sale items.", "source": "docs/sales.md"}
    ],
    "memory": [
        "The user asked about a refund yesterday."
    ],
    "messages": [
        {"role": "user", "content": "How long will my refund take?"}
    ]
}
```

### Format B: The "OpenAI" Format
If you are already formatting your messages for OpenAI, Mistral, Llama, or using libraries like LangChain, you can just paste that exact message array.

**Example (`my_prompt.json`):**
```json
[
    {"role": "system", "content": "You are a helpful customer support bot."},
    {"role": "user", "content": "How long will my refund take?"}
]
```

---

## Step 3: Analyze Your Prompt

Now that you have your `.json` file, let's see how good it is. Open your terminal and run the `inspect` command:

```bash
contextops inspect my_prompt.json
```

**What happens next?**
ContextOps will read your file and print a report. It will tell you:
1. **Your Score:** A number from 0 to 100.
2. **Token Waste:** How many tokens you are wasting (which equals wasted money).
3. **The Fix:** Exactly what you need to change (e.g., "Chunk 1 and Chunk 2 are exactly the same, delete one.")

---

## Step 4: Understanding the Penalties

If your score is below 100, it's because ContextOps caught one of four common mistakes. Here is what they mean in simple terms:

* **🔴 Redundancy:** You passed the exact same information multiple times. (E.g., giving the AI the same paragraph twice).
* **🔴 Density:** You used a high proportion of formatting overhead, whitespace, or boilerplate compared to literal payload.
* **🔴 Structure Imbalance:** You gave the AI so many background documents that it forgot what your actual instruction was! (Also known as being "Lost in the Middle").
* **🔴 Concentration:** You pulled 10 documents, but they were all from the exact same page, or the distribution is highly uneven. The AI needs balanced information to answer complex questions without source dominance.

---

## Step 5: Advanced Features for Teams

If you are working on a team or building a massive app, you don't want to check files manually every time. Here is how you automate it:

### 1. The CI/CD Quality Gate (Automated Checking)
You can tell ContextOps to "fail" if a prompt is too bad. This is perfect for GitHub Actions!

```bash
contextops check my_prompt.json --min-score 80
```
*If the score is 79 or lower, ContextOps will trigger an error. This stops bad code from ever reaching production.*

### 2. Compare Two Prompts (A/B Testing)
Did you change your search algorithm? Want to know if the new prompts are better or worse than the old ones? Use the `diff` command:

```bash
contextops diff old_prompt.json new_prompt.json
```
*This will tell you exactly if your changes improved the score or ruined it.*

---

## Summary
1. Install it: `pip install contextops`
2. Save your prompt as a `.json` file.
3. Check your score: `contextops inspect file.json`
4. Fix the errors, save money, and get smarter AI responses!
