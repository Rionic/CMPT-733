
import re
import pandas as pd
from collections import defaultdict

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)
    
        
    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 
        
        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))
        
        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))
        
        return result_df
       

    def preprocess_df(self, df, cols):
        def str_tokenizer(row):
            concat = ' '.join(row)
            tokens = [token.lower() for token in re.split(r'\W+', concat) if token]
            return tokens
        
        df['joinKey'] = df[cols].fillna('').apply(str_tokenizer, axis=1)

        return df
        
    def filtering(self, df1, df2):
        df1 = df1.rename(columns={'id': 'id1', 'joinKey': 'joinKey1'})
        df2 = df2.rename(columns={'id': 'id2', 'joinKey': 'joinKey2'})
        df1 = df1[['id1', 'joinKey1']]
        df2 = df2[['id2', 'joinKey2']]
        
        token_to_rows1 = defaultdict(list)
        token_to_rows2 = defaultdict(list)
        
        for idx, row in df1.iterrows():
            for token in row['joinKey1']:
                token_to_rows1[token].append(idx)
        
        for idx, row in df2.iterrows():
            for token in row['joinKey2']:
                token_to_rows2[token].append(idx)
        
        candidate_pairs = set()
        for token in token_to_rows1:
            if token in token_to_rows2:
                for i in token_to_rows1[token]:
                    for j in token_to_rows2[token]:
                        candidate_pairs.add((i, j))
        
        rows = []
        for i, j in candidate_pairs:
            row1 = df1.loc[i]
            row2 = df2.loc[j]
            rows.append({
                'id1': row1['id1'],
                'joinKey1': row1['joinKey1'],
                'id2': row2['id2'],
                'joinKey2': row2['joinKey2']
            })
        
        cand_df = pd.DataFrame(rows)
        return cand_df


    def verification(self, cand_df, threshold):
        def calculate_jaccard(row):
            set1 = set(row['joinKey1'])
            set2 = set(row['joinKey2'])
            return len(set1 & set2) / len(set1 | set2)
        
        cand_df['jaccard'] = cand_df.apply(calculate_jaccard, axis=1)
        
        result_df = cand_df[cand_df['jaccard'] >= threshold]
        
        return result_df
        
        
    def evaluate(self, result, ground_truth):
        result_set = set([tuple(pair) for pair in result])
        ground_truth_set = set([tuple(pair) for pair in ground_truth])

        true_positives = len(result_set & ground_truth_set)
        false_positives = len(result_set - ground_truth_set)
        false_negatives = len(ground_truth_set - result_set)
        
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        fmeasure = 2 * precision * recall / (precision + recall)
        
        return (precision, recall, fmeasure)
        
if __name__ == "__main__":
    file1 = "A2-data/part1-similarity-join/Amazon-Google-Sample/Amazon_sample.csv"
    file2 = "A2-data/part1-similarity-join/Amazon-Google-Sample/Google_sample.csv"
    er = SimilarityJoin(file1, file2)
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)
    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("A2-data/part1-similarity-join/Amazon-Google-Sample/Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))