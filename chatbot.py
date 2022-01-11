import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle



#City Checker
from geotext import GeoText


nltk.download('punkt')

with open("C:/Users/Johannes/Desktop/Chatbot/data/intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
  model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
  model.save("model.tflearn")
  
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

   
#Erstellen der Datenbank als Mockup

import pandas as pd
import numpy as np

numpy_arr = np.array ([[7234562,'Anna','Mueller','Marienstr.',12,'Duesseldorf',56789,1234,'1.3',1245,'2.3',5556],[7166521,'Markus','Shmidt','Volklingerstr.',5,'Koeln',50667,5678,'1.3',6547,'2.0',4567],[7345673,'Maria','Xi','Feldweg',2,'Duesseldorf',40210,1234,'1,7',1245,'2.0',4567],[7623451,'Philipp','Nowak','Galenstr',5,'Dortmund',44137,5678,3.0,6547,'1.0',5556 ],[7122456,'Christian','Klassen','Bachfeld',6,'Dortmund',44138,1234,4.0,1245,'1.3',5556]], dtype=object)
df = pd.DataFrame(numpy_arr, columns = ['Matriculation_number','Name','Surname','Street','House_number','City', 'ZIP', 'Passed_Exam1', 'Passed_Exam1_Grade', 'Passed_Exam2','Passed_Exam2_Grade', 'Applied_Exam' ])
print(df)

Course = np.array([[1234,'Physics'],[5678,'Economics'],[1245,'English'],[6547,'Mathematics1'],[5556,'Mathematics2'],[4567,'Object_Oriented_Programming'], [1111,'Parallel_Programming'],[1112,'Prolog_with_Applications'],[1113,'Compiler_Construction'],[1114,'Model_Driven_Software_Development']], dtype=object)
df = pd.DataFrame(Course, columns = ['ID_Subject', 'Subjects_Name'])
print(df)

   

    
def chat():
    #Wichtige Zahlen vorderfinieren als 0
    exam_no = 0
    matr_no = 0
    house_no = 0
    citycode = 0  
    city = []
    chatbotname = 'Bo: '
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")
        
        results = model.predict([bag_of_words(inp, words)])
        # Wahrscheinlichkeit für den Tag ? print(model.score(results)) ab prozentzahl in tag 
        
        results_index = numpy.argmax(results)
        tag = labels[results_index]
                
        #Wichtige Tags noch weiterer Usecases hinzufügen:
        greeting = 'greeting'
        goodbye = 'goodbye'
        change_adress = 'change_address'
        change_name = 'change_name'
        exam_reg = 'exam_reg'
        exam_dereg = 'exam_dereg'
        paid = 'paid'
        multiple_intents = 'multiple_intents'
        
        
        #Input auf vorgeschriebene Muster/Nummern untersuchen und diese Speichern. 
        house_no, exam_no, citycode, matr_no = checkingNumbers(inp,house_no, exam_no, citycode, matr_no)
        
        #Erkennen ob Stadt mit gegeben ist
        city = GeoText(inp).cities
        
        #Greeting mit reinnehmen Intents
        if tag == greeting:
            responses = data["intents"][0]["responses"]
            print(chatbotname + random.choice(responses))

        #Wann soll Chatbot ausgeschaltet werden?
        elif tag == goodbye:
            responses = data["intents"][1]["responses"]
            print(chatbotname + random.choice(responses))
            break
        
        #Behandlung Use Case Umzug
        elif tag == change_adress:
            print(city)
            print(change_adress)
            
            #Handling einführen    
                    
        elif tag == change_name:
            print(change_name)
        
        elif tag == exam_reg:
            while(matr_no == 0): #Solange User keine Matrikelnummer eingibt hier gefangen
                print(chatbotname + 'Please enter your matriculation number.')
                inp = input('You:')
                _,_,_,matr_no = checkingNumbers(inp ,house_no, exam_no, citycode, matr_no)
            while(exam_no == 0):
                print(chatbotname + 'Please enter your exam number.')
                inp = input('You:')
                _, exam_no, _, _ = checkingNumbers(inp,house_no, exam_no, citycode, matr_no)
            register_exam(matr_no, exam_no)

            
        elif tag == exam_dereg:
            while(matr_no == 0): #Solange User keine Matrikelnummer eingibt hier gefangen
                print(chatbotname + 'Please enter your matriculation number.')
                inp = input('You:')
                _,_,_,matr_no = checkingNumbers(inp ,house_no, exam_no, citycode, matr_no)
            while(exam_no == 0):
                print(chatbotname + 'Please enter your exam number.')
                inp = input('You:')
                _, exam_no, _, _ = checkingNumbers(inp,house_no, exam_no, citycode, matr_no)
            deregister_exam(matr_no, exam_no)
            
        elif tag == paid:
            if checkPaid(matr_no) == True:
                print(chatbotname + 'You dont have any open payments')
            else:
                print(chatbotname + 'Your semesterfee is not marked as paid right now.')
            
        elif tag == multiple_intents:
            print(multiple_intents)
            
        #Restliche Fälle vernünftiges Handling für schrott eingaben etc finden 
        else:
            print(chatbotname + 'Sorry i didnt understand that.')
            
def checkingNumbers(inp, house_no, exam_no, citycode, matr_no):
    inp_token = nltk.word_tokenize(inp)
    
    house_no_token = None
    
    
    for token in inp_token:
            #Checken ob Matrikelnummer, PLZ oder Prüfungsnummer in intents
            
            #Wenn Hausnummern dann ist token davor wsl -> Straße
            #Wenn PLZ ist token danach wsl -> Stadt 
            
            try:
                number = int(token)
                if(000 <= number <= 999):
                    house_no = number
                    print('housenumber')
                    print(house_no)
                    print(inp_token)
                if(1000 <= number <= 9999):
                    exam_no = number
                    print('exam')
                    print(exam_no)
                if(10000 <= number <= 99999):
                    citycode = number
                    print('city')
                    print(citycode)
                if(1000000 <= number <= 9999999):
                    matr_no = number
                    print('matr')
                    print(matr_no)
            except:
                pass
    return house_no, exam_no, citycode, matr_no
  
def register_exam(matr_no , exam_no):
    #exam no und matr no im df checken ob vorhadnen?
    #wenn ja registern wenn nein gibt kein exam / studenten
    print()
    
def deregister_exam(matr_no , exam_no):
    print('deregister')
    print('matr_no:')
    print(matr_no)
    print('exam_no:')
    print(exam_no)  
    
def changeAdress(matr_no, citycode, city, street, housenumber):
    print(citycode)

def changeName(matr_no, name, surname):
    print(matr_no)

def checkPaid(matr_no):
    return True

#Matr No noch auslagern? -> nicht null und valide ! 


chat()