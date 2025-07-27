import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV files
questions_df = pd.read_csv("Hindi.csv")  # contains Question, Answer, Domain
responses_df = pd.read_csv("Hindi_Responses_Cleaned.csv")  # contains Prompt, Response

# Merge them on Question = Prompt
merged_df = pd.merge(responses_df, questions_df, left_on='Prompt', right_on='Question', how='inner')
merged_df.columns = merged_df.columns.str.strip()

# Load sentence embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print(merged_df.columns)

# Compute similarity between Response and Answer
def compute_similarity(row):
    embeddings = model.encode([row['Response'], row['Answer']])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

merged_df['Similarity'] = merged_df.apply(compute_similarity, axis=1)

# Final columns
final_df = merged_df[['Prompt', 'Response', 'Answer', 'Similarity']]

# Save to CSV
final_df.to_csv("final_Hindi_output.csv", index=False)

print("Done. Output saved to 'final_Hindi_output.csv'")
