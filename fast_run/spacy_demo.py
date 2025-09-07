#!/usr/bin/env python3
"""
Demo of spaCy capabilities for NLP processing
"""

import spacy

# Load English model (need to install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy model: python -m spacy download en_core_web_sm")
    exit(1)

# Example text about 10-K filings
text = """Apple Inc. reported revenue of $394.3 billion in fiscal year 2023. 
Tim Cook, the CEO, mentioned that the company faces competition from Microsoft 
and Google in the artificial intelligence market. The Cupertino-based company 
invested heavily in machine learning research and development."""

print("🔤 SPACY NLP DEMO")
print("=" * 50)
print(f"Text: {text}")
print("=" * 50)

# Process the text
doc = nlp(text)

# 1. Named Entity Recognition
print("\n🏷️  NAMED ENTITIES:")
for ent in doc.ents:
    print(f"   {ent.text:<20} | {ent.label_:<10} | {spacy.explain(ent.label_)}")

# 2. Part-of-speech tagging
print("\n📝 PART-OF-SPEECH TAGS (first 10 tokens):")
for token in doc[:10]:
    print(f"   {token.text:<15} | {token.pos_:<8} | {spacy.explain(token.pos_)}")

# 3. Dependency parsing
print("\n🔗 DEPENDENCY RELATIONSHIPS (key ones):")
for token in doc:
    if token.dep_ in ['nsubj', 'dobj', 'ROOT']:
        print(f"   {token.text:<15} | {token.dep_:<8} | {token.head.text}")

# 4. Sentence segmentation
print(f"\n📄 SENTENCES:")
for i, sent in enumerate(doc.sents, 1):
    print(f"   {i}. {sent.text.strip()}")

# 5. Lemmatization
print(f"\n🔤 LEMMATIZATION (verbs and nouns):")
for token in doc:
    if token.pos_ in ['VERB', 'NOUN'] and token.text != token.lemma_:
        print(f"   {token.text} → {token.lemma_}")

print("\n" + "=" * 50)
print("✅ spaCy demo complete!")
