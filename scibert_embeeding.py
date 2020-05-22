#previously working liene
import flask
import pickle
import torch

from transformers import *
#from transformers import AutoTokenizer, AutoModelWithLMHead # poor performance with bog and flu

#model_version = 'scibert_scivocab_uncased'
#model = BertModel.from_pretrained(model_version)
#model = AutoModelWithLMHead.from_pretrained("deepset/covid_bert_base") # poorer initial search result effectiveness

do_lower_case = True
#tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
#tokenizer = AutoTokenizer.from_pretrained("deepset/covid_bert_base")

from sklearn.metrics.pairwise import cosine_similarity

#def embed_text(text, model):
def embed_text(text):
    #input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    input_ids = torch.tensor(BertTokenizer.from_pretrained('scibert_scivocab_uncased', do_lower_case=True).encode(text)).unsqueeze(0)  # Batch size 1
    #outputs = model(input_ids)
    outputs = BertModel.from_pretrained('scibert_scivocab_uncased')(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states

def get_similarity(em, em2):
    return cosine_similarity(em.detach().numpy(), em2.detach().numpy())

# We will use a mean of all word embeddings. To do that we will take mean over dimension 1 which is the sequence length.
#coronavirus_em = embed_text("Coronavirus", model).mean(1)
coronavirus_em = embed_text("Coronavirus").mean(1)
#mers_em = embed_text("Middle East Respiratory Virus", model).mean(1)
mers_em = embed_text("Middle East Respiratory Virus").mean(1)
#flu_em = embed_text("Flu", model).mean(1)
flu_em = embed_text("Flu").mean(1)
#bog_em = embed_text("Bog", model).mean(1)
bog_em = embed_text("Bog").mean(1)
#covid_2019 = embed_text("COVID-2019", model).mean(1)
covid_2019 = embed_text("COVID-2019").mean(1)
print("Similarity for Coronavirus and Flu:" + str(get_similarity(coronavirus_em, flu_em)))
print("Similarity for Coronavirus and MERs:" + str(get_similarity(coronavirus_em, mers_em)))
print("Similarity for Coronavirus and COVID-2019:" + str(get_similarity(coronavirus_em, covid_2019)))
print("Similarity for Coronavirus and Bog:" + str(get_similarity(coronavirus_em, bog_em)))

import pandas as pd

def make_the_embeds(number_files, start_range=0,
                    data_key=["title"]):
    df = pd.read_csv('data/covid_pdf.csv')
    #the_list = os.listdir(the_path)
    title_embedding_list = []
    title_list = []
    for i in range(start_range, number_files):
        #file_name = the_list[i]
        #final_path = os.path.join(the_path, file_name)
        #with open(final_path) as f:
        #    data = json.load(f)
        try:
            tensor, title = make_data_embedding(df.loc[i], data_key)
            #print('title: ')
            #print(title)
            #print('tensor: ')
            #print(tensor)
            title_embedding_list.append(tensor)
            title_list.append(title)
        except:
            print("Invalid title/abstract")
    return torch.cat(title_embedding_list, dim=0), title_list


def make_data_embedding(article_data, data_keys, method="mean", dim=1):
    text = embed_text(article_data[data_keys], model)
    if method == "mean":
        return text.mean(dim), article_data[data_keys]


#embed_list, title_list = make_the_embeds(200, 0, 'title') # parse 200 due to kaggle/platform restrictions
embed_list, title_list = make_the_embeds(200, 0, 'title') # parse 100, try to reduce heroku slug size
#red = reducer.fit_transform(embed_list.detach().numpy())  #
print('title_list: ')
print(title_list)

#embed_list2, title_list2 = make_the_embeds(401, 201, 'title') # parse 200 due to kaggle/platform restrictions
embed_list2, title_list2 = make_the_embeds(201, 101, 'title') # parse 100, try to reduce heroku slug size
print('title_list2: ')
print(title_list2)

from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')


import collections
q1 = "COVID-19 infection origin and transmission from animals"
#search_terms = embed_text(q1, model).mean(1)
search_terms = embed_text(q1).mean(1)

def top_n_closest(search_term_embedding, title_embeddings, original_titles, n=10):
    proximity_dict = {}
    i = 0
    for title_embedding in title_embeddings:
        proximity_dict[original_titles[i]] = {"score": get_similarity(title_embedding.unsqueeze(0),search_term_embedding),
                                              "title_embedding":title_embedding.unsqueeze(0)}
        i+=1

    order_dict = collections.OrderedDict({k: v for k, v in sorted(proximity_dict.items(), key=lambda item: item[1]["score"])})
    proper_list = list(order_dict.keys())[-n:]
    return proper_list, order_dict


top_titles, order_dict = top_n_closest(search_terms, embed_list2, title_list+title_list2)
print(top_titles)


# Use pickle to load in the pre-trained model.
with open(f'model/SciBert_TopTitles.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        windspeed = flask.request.form['windspeed']
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       columns=['temperature', 'humidity', 'windspeed'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Temperature':temperature,
                                                     'Humidity':humidity,
                                                     'Windspeed':windspeed},
                                     result=prediction,
                                     )