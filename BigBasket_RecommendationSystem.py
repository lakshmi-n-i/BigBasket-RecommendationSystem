import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

listings = pd.read_csv("bigbasket.csv", usecols = ['product', 'category', 'description'])
listings.head(10)

listings['product']=listings['product'].astype('str')
listings['description']=listings['description'].astype('str')
listings['category']=listings['category'].astype('str')

#.astype() is used to change the datatype to str (in case some columns have integer values

listings['content']=listings[['product','category','description']].astype(str).apply(lambda x:'//'.join(x),axis=1)

"""
axis =0 applies to each column *and*
axis=1 applies to each row

.apply() is to perform a function on the DataFrame
"""

listings['content']

listings['content'].fillna('Null',inplace=True)

#Replaces all null values with "NULL"

#---------------------------------------------------------
#Training the recommender: TfidfVectorizer() converts a collection of raw documents into a matrix of TF-IDF feature
#---------------------------------------------------------------

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(listings['content'])

"""
analyzer: extract the sequence of words out of the raw, unprocessed input.

ngram_range: defines the range of different n-grams to be extracted.(1,2) means unigrams and bigrams are to be extracted.

min_df=is used for removing terms that appear too infrequently.

"""

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

#Cosine similarity: contains the pairwise cosine similarity score for every pair of entries(or their vectors). If there are 'n' entries, a (n*n) matrix is formed, where the value corresponding to the ith row and jth column(i.e, index=[i][j]) denotes the similarity score for the ith and jth vector.

#both linear_kernel() and cosine_similarity() produce the same result but linear_kernel() executes faster and is more efficient for a larger dataset

results = {}
for idx, row in listings.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], listings['product'][i]) for i in similar_indices]
    results[row['product']] = similar_items[1:]

"""
Iterate through each item's similar items and store the 100 most-similar

“idx” means index, Dataframe.iterrows() specifically knows that idx is index.

DataFrame.iterrows() allows us to iterate each index and row in the DataFrame. Each iteration produces an index object and a row object

.argsort() returns an array of indices that sort the given array. It doesn't return the sorted array, but the array of indexes which indicate the sorted positions.

"similar_indices" is a 2D array/matrix having the cosine similarities of each row, with every other row in the dataset.  Here, .argsort() sorts each row (the cosine similarities) and represents the sorted list in terms of their index positions

list slicing[:-100:-1] is used to get the last 100 cosine similarities( which will invariably be the values closest to 1). Thus, these are the values that are most similar to the product given by user
"""

similar_indices

#similar_indices is giving the sorted order of the lisgt, in terms of their respective index positions, not the values.It stores the top 100 closest values for each row/product in the dataset

similar_items

"""
similar_items is a nested list, where each nested list item has 2 values: the cosine similarity and its corresponding product name

For each entry(cosine similarity) in similar_indices, the nested list "similar_items" stores the same cosine similarity, along with its corresponding product name. It stores 100 pairs of values, for each row in the dataset. Each pair consists of the cosine similarity and the corresponding product name
"""

results

#results" maps each product in the dataset, to its corresponding top 100 closest values, i.e (100 pairs: each pair consists of cosine similarity and corresponding product name). It maps each product in the dataset, to the corresponding entry in "similar_items"

#Prediction
def item(id):
    name   = listings.loc[listings['product'] == id]['content'].tolist()[0].split('//')[0]
    desc   = ' \nDescription: ' + listings.loc[listings['product'] == id]['content'].tolist()[0].split('//')[2][0:165] + '...'
    prediction = name  + desc
    return prediction

"""
.loc() is the function used for filtering "listings" DataFrame to get only those rows where the "product" column matches the id parameter. It then extracts the "content" column from that filtered DataFrame and converts it to list. Conversion to lists is done only because lists are way easier to work with

The [0] index is used to access the first item of the list, which is the value of the "content" column. Thus we get "product//category//description" format

The value from ".tolist()[0]" (i.e, the value of the "content" column) is split around "//". The function .split() returns a list of the substrings

The [0] index of the list returned by .split() consists of the product name.

The [1] index of the list returned by .split() consists of the category.

The [2] index of the list returned by .split() consists of the description.

The [0:165] is to extract the first 165 characters of the description. This is for presentation purpose only
"""

def recommend(product, num):
    print('Recommending ' + str(num) + ' products similar to ' + item(product))
    print('---')
    recs = results[product][:num]
    for rec in recs:
        print('\nRecommended: ' + item(rec[1]) + '\n(score:' + str(rec[0]) + ')')

"""
recs is storing the first "num" values/recommendations for the product. "product" is the argument taken from user input. Assume num=5 for further explanations

"for rec in recs" iterates through each of the 5 recommendations extracted, for the product(which is given by user). Each of these recommendations is a list having 2 values: cosine similarity and product name

rec[1] is the product name of the recommended product

rec[0] is the cosine similarity value of the recommended product

# OVERALL
item() is used to extract name and description column values of a particular product

recommend() calls the function item() for each entry of the selected set of recommended products. These recommended products are extracted by using cosine simlarity and filtering out the closest matches
"""

recommend(product = "Masala Upma", num = 5)