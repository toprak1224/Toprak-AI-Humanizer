Toprak AI Humanizer
An advanced desktop application designed to transform AI-generated text into more natural, sophisticated, and human-like prose.

Toprak AI Humanizer is more than a simple paraphrasing tool. It's a sophisticated suite of NLP models and techniques packaged in a modern, user-friendly interface built with PySide6. It intelligently analyzes and refines text to help bypass AI detection systems and enhance overall readability.

<img width="1396" height="825" alt="image" src="https://github.com/user-attachments/assets/c4c29c1d-ffe0-4e80-9cf3-ddd3b331c863" />

✨ Key Features
This tool leverages a multi-layered approach to text transformation:

🧠 Deep Paraphrasing: Utilizes a T5 transformer model to rewrite sentences from scratch, changing their structure while preserving the original meaning.

✍️ Intelligent Synonym Replacement: Swaps words with context-aware synonyms using Sentence Transformers, ensuring the new words fit the sentence's context.

🛡️ Sentiment Guard: A crucial layer that performs sentiment analysis with TextBlob to prevent word swaps that would reverse the meaning (e.g., replacing a positive word like "impact" with a negative one like "encroachment").

🛠️ Automatic Grammar Correction: After modifications, it runs the text through a grammar correction engine (language-tool-python) to fix subject-verb agreement, word forms, and other grammatical errors.

🔄 Sentence Restructuring: Varies the rhythm of the text by combining short sentences and splitting long, complex ones to create a more natural, human-like flow.

🌐 Bilingual Interface: Supports both English and Turkish, with the ability to switch languages on the fly.

🎨 Modern UI: A clean, dark-themed, and responsive user interface built with PySide6, providing real-time status updates during processing.

🛠️ Tech Stack
Backend: Python

GUI: PySide6

Core NLP Models:

Hugging Face Transformers (for T5 Paraphrasing)

Sentence-Transformers (for Contextual Embeddings)

spaCy (for core NLP tasks like sentence segmentation)

NLTK (for tokenization and POS tagging)

Enhancement Layers:

language-tool-python (for Grammar Correction)

textblob (for Sentiment Analysis)

🚀 Installation
Follow these steps to get the application running on your local machine.

Prerequisites
Python 3.9+

Java Runtime Environment (JRE): The "Grammar Correction" feature depends on language-tool-python, which requires Java to be installed on your system.
