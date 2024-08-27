
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd

df = pd.read_csv("immo_data.csv")

df = df.drop_duplicates( "scoutId" , keep='first')

df = df[df['description'].notna()]
df = df[df['facilities'].notna()]
df = df[df['description'].apply(lambda x: len(x.split()) <= 300)]
df = df[df['facilities'].apply(lambda x: len(x.split()) <= 300)]

print('Dataframe size : ', len(df))

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

client = chromadb.PersistentClient(path="chromadb_vector_storage")

def vectorize_and_store_embeddings(collection_name):
    collection = client.create_collection(collection_name)

    print(collection_name, ' is created.')
    for i in range(0, len(df), 100):

        ids = []
        sentences = []
        for j in range(i, i+100):

            if j == len(df): break

            row = df.iloc[j]
            ids.append(str(row['scoutId']))
            sentences.append(row[collection_name])

        embeddings = model.encode(sentences)
        collection.add(ids=ids, embeddings=embeddings.tolist(), documents=sentences)

        print(collection_name, ' : ', str(i), ' / ', str(len(df)))
    print(collection_name, ' is done.')


if __name__=='__main__':
    vectorize_and_store_embeddings('description')
    vectorize_and_store_embeddings('facilities')

