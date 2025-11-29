# #--------------------------------
# # Example: Remove documents that overlap with MMLU/HellaSwag/etc.

# benchmark_texts = load_benchmark_data()  # Load all eval datasets

# def has_contamination(doc, min_ngram=13):
#     """Check if doc contains long n-grams from benchmarks"""
#     doc_ngrams = set(get_ngrams(doc, min_ngram))

#     for bench_text in benchmark_texts:
#         bench_ngrams = set(get_ngrams(bench_text, min_ngram))
#         if doc_ngrams & bench_ngrams:  # Any overlap
#             return True
#     return False

# clean_data = [doc for doc in data if not has_contamination(doc)]

# #--------------------------------
# #Don't let one domain dominate the dataset:

# from collections import Counter

# # Count documents by domain
# domain_counts = Counter(doc['domain'] for doc in data)

# # Downsample over-represented domains
# target_max = 1_000_000  # Max docs per domain

# balanced_data = []
# domain_seen = Counter()

# for doc in shuffled(data):
#     domain = doc['domain']
#     if domain_seen[domain] < target_max:
#         balanced_data.append(doc)
#         domain_seen[domain] += 1

# #--------------------------------
# # Example: Remove common navigation text
# # This is to avoid the model from learning to navigate the web.
# boilerplate_phrases = [
#     "Click here to subscribe",
#     "All rights reserved",
#     "Cookie policy",
#     "Share on Facebook",
#     # ... add more
# ]

# def remove_boilerplate(text):
#     for phrase in boilerplate_phrases:
#         text = text.replace(phrase, "")
#     return text
