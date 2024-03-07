import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Embedding, LSTM, Dense, Dropout, Input
from keras.layers.merge import concatenate
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from functools import partial
from keras.metrics import binary_crossentropy
from skmultilearn.model_selection import iterative_train_test_split
from keras.utils.vis_utils import plot_model

#set the random seed for tensorflow
tf.random.set_seed(42)

def weightedCrossentropy(yTrue, yPred, weight):
  """Given predictions and a weight, weight the binary cross entrophy score.
  Parameters:
    yTrue - the true outcomes.
    yPred - the predicted outcomes.
    weight - the weight to be used.

  Return:
    bce - the weighted binary cross entrophy score.
  """
  bce = binary_crossentropy(yTrue,yPred) * weight
  return bce

def createCombinedModel(inputLenTitles, inputLenAbstracts, outputLen, weights):
  """Given dimesions and weights, create an LSTM neural network.

  Parameters:
    inputLenTitles - the number of words in the title vocabulary.
    inputLenAbstracts - the number of words in the abstract vocabulary.
    outputLen - the number of binary classes the model is predicting.
    weights - the weights of each binary class.

  Return:
    model - the compiled ANN.
  """
  #the title branch of the ANN
  visible1 = Input(shape=(inputLenTitles,), name='Title_Inputs')
  em1 = Embedding(inputLenTitles, 20, input_length=inputLenTitles, name="Title_Embedding")(visible1)
  lstm1 = LSTM(40,return_sequences=True, name='Title_LSTM1')(em1)
  drop1 = Dropout(0.2, name='Title_Dropout1')(lstm1)
  lstm11 = LSTM(20, name='Title_LSTM2')(drop1)
  drop11 = Dropout(0.1, name='Title_Dropout2')(lstm11)
  
  #the abstract branch of the ANN
  visible2 = Input(shape=(inputLenAbstracts,), name='Abstract_Inputs')
  em2 = Embedding(inputLenAbstracts, 20, input_length=inputLenAbstracts, name='Abstract_Embedding')(visible2)
  lstm2 = LSTM(40, return_sequences=True, name='Abstract_LSTM1')(em2)
  drop2 = Dropout(0.2, name='Abstract_Dropout1')(lstm2)
  lstm22 = LSTM(20, name='Abstract_LSTM2')(drop2)
  drop22 = Dropout(0.1, name='Abstract_Dropout2')(lstm22)

  #merge the two branches and get outputs
  merge = concatenate([drop11, drop22], name='Title_and_Abstract_Merge')
  dense3 = Dense(10, activation='relu', name='Dense_Layer1')(merge)
  output = Dense(outputLen, activation='sigmoid', name='Keyword_Output')(dense3)

  #where losses will be stored
  losses = []

  #generate the losses given the class weights
  for i in weights.keys():
     losses.append(partial(weightedCrossentropy, weight=weights[i]))

  #create and compile the whole model
  model = Model(inputs=[visible1, visible2], outputs=output)
  model.compile(loss=losses, optimizer='Adam',metrics=['accuracy'])

  plot_model(model, to_file='model.png')

  return model

def tokenize(text,numWords,filter='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
  """Given a list of lists, tokenize the lists to contain a certain number of
  words, convert the list of lists to a numpy matrix and return it. 

  Parameters:
    text - sentences which contain words to be tokenized.
    numWords - the number of words to retain in tokenization.
    filter - filters

  Return:
    encoded - a binary encoded matrix that shows the representation of certain words in a sentence.
  """
  #create the tokenizer, fit to the text and then transform the text to a matrix
  t = Tokenizer(num_words=numWords, filters=filter)
  t.fit_on_texts(text)
  encoded = t.texts_to_matrix(text, mode='binary')
  return encoded

def filterSequence(df):
  """Given a series of sentences, remove unneeded words and convert into a list of words.

  Parameters:
    df - the series of sentences.

  Return:
    df - the adjusted series, which has removed filler words and converted each sentence into a list.
  """
  #words to not be included in the sequence
  fillerWords = ['for','with','using','on','of','an','by','in','and','via','the','a','to',
                'these','us','can','used','where','when','who','what','how','against','given','both']
  
  #make all words lowercase, remove filler words, and convert sentences to list of words
  df = df.str.lower()
  df = df.str.replace(r'\b{}\b'.format('|'.join(fillerWords)), '', regex=True)
  df = df.apply(lambda x: text_to_word_sequence(x))

  return df


def predictKeywords():
  """Predict keywords/terms from the arxiv dataset."""

  #read in the dataset
  df = pd.read_csv("dataset/arxiv_data.csv")

  #number of words to use in vocab of titles, abstracts, and keywords
  titleSize = 41
  abstractSize = 41
  keywordSize = 4

  #clean and sequence the titles and abstracts
  df['titles'] = filterSequence(df['titles'])
  df['summaries'] = filterSequence(df['summaries'])

  #terms has brackets so it must be cleaned differently
  df['terms'] = df['terms'].apply(lambda x: x.replace("'",""))
  df['terms'] = df['terms'].apply(lambda x: x.replace(" ",""))
  df['terms'] = df['terms'].apply(lambda x: x.lower())
  df['terms'] = df['terms'].apply(lambda x: x.strip('[]').split(','))

  #get titles, abstracts, and keywords as lists
  titles = df['titles'].to_list()
  abstracts = df['summaries'].to_list()
  terms = df['terms'].to_list()

  #tokenize and encode the titles, abstracts, and keywords
  encodedTitles = tokenize(titles,titleSize)
  encodedAbstracts = tokenize(abstracts,abstractSize)
  encodedTerms = tokenize(terms,keywordSize,filter="[]''")

  #join titles and abstracts horizontally for the train test split
  jointArray = np.hstack((encodedTitles, encodedAbstracts))
  
  #split into 80% training and 20% testing accounting for class representation
  X_train, y_train, X_test, y_test = iterative_train_test_split(jointArray,encodedTerms, test_size=0.20)

  #set the titles and abstracts training/testing columns
  titles_train =  X_train[:,:titleSize]
  abstracts_train =  X_train[:,-abstractSize:]
  titles_test =  X_test[:,:titleSize]
  abstracts_test =  X_test[:,-abstractSize:]

  #weight classes based on representation in the dataset
  class_series = np.argmax(encodedTerms, axis=1)
  class_labels = np.unique(class_series)
  class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
  weights = dict(zip(class_labels, class_weights))

  #create a fit the model on the training data
  model = createCombinedModel(titles_train.shape[1],abstracts_train.shape[1],y_train.shape[1],weights)
  model.fit([titles_train,abstracts_train],y_train,epochs=25) 

  #predict the outcomes from the testing set
  preds = model.predict([titles_test,abstracts_test])
  preds = np.array([[int(np.round(float(i))) for i in nested] for nested in preds])

  print(classification_report(np.array(y_test), preds))

predictKeywords()