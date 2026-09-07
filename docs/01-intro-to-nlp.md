# 1. Introduction to NLP

## What is NLP?

Natural Language Processing (NLP) is a subfield of Artificial Intelligence (AI) that focuses on enabling computers to understand, interpret, generate, and manipulate human language (text and speech) in a way that is meaningful and useful.

It sits at the intersection of:

* **Linguistics** – structure and meaning of language
* **Computer Science** – algorithms and computation
* **Machine Learning / AI** – learning patterns from data

**[Diagram placeholder: Venn diagram showing NLP at intersection of Linguistics, Computer Science, and AI]**

---

## Need for NLP

* Massive amount of unstructured text/speech data is generated daily (emails, social media, reviews, articles, chats).
* Humans cannot manually process this data at scale.
* Businesses need to extract insights, automate communication, and improve decision-making from language data.
* Enables human-computer interaction using natural language instead of rigid commands/code.

---

## What exactly is Natural Language?

* Natural language refers to languages that evolved naturally through human use (e.g., English, Hindi, Spanish) — as opposed to **constructed/formal languages** (e.g., programming languages, mathematical notation).
* Characteristics of natural language:

  * **Ambiguous** (same word/sentence can have multiple meanings)
  * **Evolving** (new words, slang, usage patterns emerge)
  * **Context-dependent** (meaning changes based on context)
  * **Redundant/flexible in structure** (same meaning expressed in many ways)

---

## Real World Applications of NLP

| Application                   | Description                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| **Contextual Advertisements** | Ads served based on the meaning/context of content being viewed (e.g., Google AdSense) |
| **Email Clients**             | Spam filtering, smart reply, priority inbox (e.g., Gmail)                              |
| **Social Media**              | Trend detection, content moderation, sentiment tracking                                |
| **Search Engines**            | Understanding query intent, ranking relevant documents (e.g., Google Search)           |
| **Chatbots**                  | Automated conversational agents for customer support, assistants (e.g., Siri, Alexa)   |

---

## Common NLP Tasks

1. **Text/Document Classification** – Assigning predefined categories to text (e.g., spam vs. not spam)
2. **Sentiment Analysis** – Determining emotional tone (positive/negative/neutral)
3. **Information Retrieval** – Finding relevant documents/information from a large corpus (e.g., search engines)
4. **Parts of Speech (POS) Tagging** – Labeling words with grammatical categories (noun, verb, adjective, etc.)
5. **Language Detection & Machine Translation** – Identifying language of text; translating text between languages
6. **Conversational Agents** – Chatbots and voice assistants
7. **Knowledge Graph & QA Systems** – Structuring knowledge as entities/relationships; answering questions from it
8. **Text Summarization** – Condensing long text into a short, meaningful summary
9. **Topic Modelling** – Discovering abstract topics within a collection of documents
10. **Text Generation** – Producing coherent, human-like text
11. **Spell Checking & Grammatical Error Correction** – Detecting/correcting spelling and grammar issues
12. **Text Parsing** – Analyzing grammatical structure of sentences
13. **Speech to Text** – Converting spoken language into written text

---

## Approaches to NLP

### 1. Heuristic Methods (Rule-Based)

Based on hand-crafted rules and lexical resources.

* **Regular Expressions (Regex)** – Pattern matching in text
* **WordNet** – Lexical database of English; groups words into sets of synonyms (synsets) with semantic relationships
* **Open Mind Common Sense** – Crowdsourced knowledge base of everyday common-sense facts

**Example: Regex for email extraction**

```python
import re

text = "Contact us at support@example.com or sales@company.org"

pattern = r'[\w.-]+@[\w.-]+\.\w+'

emails = re.findall(pattern, text)

print(emails)
# Output: ['support@example.com', 'sales@company.org']
```

---

### 2. Machine Learning Based Methods

Learn patterns from labeled/unlabeled data instead of hand-crafted rules.

**Advantages:**

* Generalizes better than fixed rules
* Handles variability and unseen patterns
* Scales with more data

**Common Algorithms:**

* **Naive Bayes** – Probabilistic classifier based on Bayes' theorem, assumes feature independence
* **Logistic Regression** – Linear model for classification tasks
* **SVM (Support Vector Machine)** – Finds optimal hyperplane to separate classes
* **LDA (Latent Dirichlet Allocation)** – Probabilistic model for topic modelling
* **Hidden Markov Models (HMM)** – Sequence modelling (e.g., POS tagging) using states and transition probabilities

**Example: Naive Bayes text classification**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "I love this movie",
    "This film was terrible",
    "Amazing acting",
    "Worst movie ever"
]

labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

test = vectorizer.transform(["I really loved it"])

print(model.predict(test))
# Output: [1]
```

---

### 3. Deep Learning Based Methods

Use neural networks to automatically learn feature representations from raw text.

**Advantages:**

* No manual feature engineering
* Captures complex, long-range patterns and context
* State-of-the-art performance on most NLP tasks

**Common Architectures:**

* **RNN (Recurrent Neural Network)** – Processes sequential data, maintains hidden state across time steps
* **LSTM (Long Short-Term Memory)** – RNN variant that handles long-term dependencies and helps address the vanishing gradient problem
* **GRU (Gated Recurrent Unit) / CNN** – GRU is a simplified LSTM; CNNs are used for capturing local patterns (n-grams) in text
* **Transformers** – Attention-based architecture that processes sequences in parallel (basis of BERT, GPT, etc.)
* **Autoencoders** – Learn compressed representations of text via encoder-decoder structure

**Example: Simple LSTM for text classification (Keras)**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

---

## Challenges in NLP

* **Ambiguity** – Lexical (multiple meanings of a word), syntactic (multiple parse structures), and semantic ambiguity
* **Context Dependency** – Same phrase can mean different things in different contexts
* **Sarcasm & Irony** – Hard to detect literal vs. intended meaning
* **Multilingualism** – Handling multiple languages, dialects, and code-switching
* **Lack of Labeled Data** – Especially for low-resource languages/domains
* **Coreference Resolution** – Identifying what pronouns/references point to
* **Domain Adaptation** – Models trained on one domain may not generalize to another
* **Evolving Language** – New slang, abbreviations, and usage patterns
* **Computational Cost** – Deep learning models require significant resources
* **Bias in Data** – Models can inherit/amplify societal biases present in training data
