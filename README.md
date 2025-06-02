---
title: Healthcare Reimbursement Agent
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app/app.py
pinned: false
---

# 🏥 Alles Health AI – Reimbursement Assistant

Alles Health AI is a **Streamlit-based AI assistant** designed to help hospital staff and administrators analyze reimbursement-related queries from complex **Tariff Documents** (PDF, Word, Excel, CSV, or JSON). It uses **Llama 3 via Groq API** with **vector search** to provide accurate and concise answers, and offers **PDF download of responses** for record-keeping.

---

## 🚀 Features

* 📁 Upload tariff documents in multiple formats (PDF, DOCX, XLSX, CSV, JSON)
* 🧠 Extracts key medical information using `langchain` + `FAISS` vector DB
* 🤖 Answers queries with Groq’s blazing-fast LLaMA 3 model
* 📝 Downloadable PDF summaries of answers with context
* 💡 Easy UI for non-technical users via Streamlit

---

## 📸 Example Test Queries

> (See screenshots in the `/examples` folder or refer to this sample image)

Here are some sample queries you can test:

| 📄 Document Content                              | 💬 Example Queries                        |
| ------------------------------------------------ | ----------------------------------------- |
| Includes procedures like Dialysis, MRI, Surgery  | `What is the reimbursement for dialysis?` |
| Contains hospital stay and room category charges | `What are the ICU charges per day?`       |
| Includes diagnostic or test codes and fees       | `What is the cost for MRI with contrast?` |

---

## ⚙️ How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/alles-health-ai.git
cd alles-health-ai
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set your API key**
   Create a `.env` file in the root with:

```
GROQ_API_KEY=your_groq_api_key
```

> Get your free API key from [https://console.groq.com](https://console.groq.com)

5. **Run the app**

```bash
streamlit run app.py
```

---

## 📦 Project Structure

```
├── app.py                    # Streamlit app
├── utils.py                  # File parsers for PDF, Excel, etc.
├── requirements.txt
├── .env                      # Your GROQ_API_KEY
├── /examples                 # Sample screenshots
```

---

## 🛠️ Trade-offs and Improvements

### ✅ Trade-offs:

* Using `all-MiniLM-L6-v2` embeddings ensures low latency but may not capture deep domain semantics.
* Groq LLaMA 3 (8B) model is fast and free, but not fine-tuned on medical data.
* Only supports single-file input at a time (no multi-document analysis yet).

### 🚀 Future Improvements:

* Add multi-document support with combined vector stores.
* Upgrade to domain-specific medical embeddings (e.g., BioBERT).
* Allow user to select answer length, chunk size, and context window dynamically.
* Implement role-based access if deployed in hospitals.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Groq API](https://groq.com/)
* [Streamlit](https://streamlit.io/)
* Inspired by real-world reimbursement challenges in healthcare.


