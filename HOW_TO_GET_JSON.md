# How to Get Your Context JSON File

A very common question when starting with ContextOps is: *"Where do I actually get the `.json` file to test?"*

You don't start with a JSON file—you capture it from your running application right before it gets sent to the AI. Here are the three most common ways developers capture their context:

## 1. The "Print & Save" Method (During Local Development)
When you are writing the code that talks to OpenAI (or any other provider), there is a moment *right before* you hit the API where you assemble the full prompt. To use ContextOps, simply add two lines of code to save that prompt to a file.

**Example Python Code:**
```python
# Your normal code assembling the prompt
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query_plus_rag_chunks}
]

# 👉 ADD THESE TWO LINES TO CAPTURE THE JSON
import json
with open("my_prompt.json", "w") as f:
    json.dump(messages, f, indent=2)

# Normal API call continues...
response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
```
Run your code once, and `my_prompt.json` will magically appear in your folder. You can now run `contextops inspect my_prompt.json` on it!

## 2. Exporting from Observability Tools (For Production Data)
If your application is already live, you are probably using a monitoring tool like **LangSmith**, **Helicone**, **Datadog**, or **Phoenix**. 

When a user reports a bad response, you don't need to guess what happened:
1. Open your observability dashboard.
2. Find the specific bad API call.
3. Click **"Export as JSON"** or **"Copy Raw Payload"**.
4. Save that payload as a `.json` file on your computer.
5. Run ContextOps on it to figure out exactly why the context was flawed.

## 3. Automated Test Suites (For CI/CD)
When writing automated tests for a RAG pipeline, developers generate prompts dynamically. 
Instead of sending them all to OpenAI and paying for test tokens, you can save them locally:

1. Have your test script run 50 different user questions.
2. Retrieve the documents for each question.
3. Save each final payload into a folder (e.g., `/test_prompts/case_1.json`, `/test_prompts/case_2.json`).
4. In your CI pipeline, run ContextOps to grade them all at once:
   ```bash
   contextops check /test_prompts/*.json --min-score 80
   ```

By capturing the exact payload you were about to send to the AI, ContextOps gives you a 100% accurate score of your context quality.
