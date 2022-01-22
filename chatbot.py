from sys import platform

unix = "darwin" in platform or "linux" in platform

from logger import Logger

Logger(level=2, debug_mode=False)

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from pandasql import sqldf

import re

nltk.download('punkt')

if unix:
    from address_processor import AddressProcessor


chatbotname = 'Bo'


def chatbot_out(*vars):
    message = " ".join(list(vars))
    print(f'{chatbotname}:', message)


def user_in():
    return input('You: ')


def get_full_name_by_matriculation_number(matriculation_number):
    Logger.debug(1, 'Get firstname by matriculation number:', matriculation_number)
    candidate_indices = student_entries_df.index[student_entries_df['Matriculation_number'] == str(matriculation_number)]
    Logger.debug(1, 'Candidate indices:', list(candidate_indices))
    firstname = student_entries_df.iloc[candidate_indices[0]]['Name']
    Logger.debug(1, 'Got firstname:', firstname)
    surname = student_entries_df.iloc[candidate_indices[0]]['Surname']
    Logger.debug(1, 'Got surname:', surname)
    return f'{firstname} {surname}'


def match_matriculation_number_from_input(input):
    Logger.debug(1, 'Search for matriculation number in input')
    regex = r'(?<!\d)\d{7}(?!\d)'
    Logger.debug(1, 'Searching RegEx in input:', regex, input)
    return re.search(regex, input)


def matriculation_number_exists(matriculation_number):
    Logger.debug(1, 'Check if matriculation number exists:', matriculation_number)
    candidate_indices = student_entries_df.index[
        student_entries_df['Matriculation_number'] == str(matriculation_number)]
    Logger.debug(1, 'Indices for given matriculation number:', list(candidate_indices))
    return len(candidate_indices) > 0


def check_for_matriculation_number(inp):
    matriculation_number = 0
    match = match_matriculation_number_from_input(inp)
    if match:
        if matriculation_number_exists(match.group()):
            matriculation_number = match.group()
        else:
            chatbot_out('I could not find a student with the given matriculation number.')

    else:
        chatbot_out('Please enter your matriculation number to continue.')
        match = match_matriculation_number_from_input(user_in())
        if match:
            if matriculation_number_exists(match.group()):
                matriculation_number = match.group()
            if not matriculation_number_exists(matriculation_number):
                chatbot_out('I could not find a student with the given matriculation number.')

        retries = 0
        while retries < 2 and not matriculation_number:
            retries += 1
            chatbot_out('Excuse me, I could not recognize your matriculation number.')
            chatbot_out('Please try again.')

            match = match_matriculation_number_from_input(user_in())
            if match:
                if matriculation_number_exists(match.group()):
                    matriculation_number = match.group()
                if not matriculation_number_exists(matriculation_number):
                    chatbot_out('I could not find a student with the given matriculation number.')

        if retries >= 2 and not matriculation_number:
            chatbot_out('I am having problems recognizing your matriculation number.')
            chatbot_out('Please choose a different action.')

    return matriculation_number


with open("data/intents.json") as file:
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

    exclusions = ["?", ",", ".", "{", "}", ":"]

    words = [stemmer.stem(w.lower()) for w in words if w not in exclusions]
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

input_layer_size = len(training[0])
hidden_layer_size = round(input_layer_size * (2/3) + len(output[0]))

input_layer = tflearn.input_data(shape=[None, input_layer_size])

hidden_layer = tflearn.fully_connected(input_layer, hidden_layer_size, activation="relu")
hidden_layer = tflearn.fully_connected(hidden_layer, hidden_layer_size, activation="sigmoid")

output_layer = tflearn.fully_connected(hidden_layer, len(output[0]), activation="softmax")

net = tflearn.regression(output_layer, metric=tflearn.metrics.Accuracy(), optimizer="rmsprop", loss="mean_square")

model = tflearn.DNN(net, tensorboard_verbose=3)

try:
    with open("model.tflearn") as chkpt:
        model.load(chkpt)
except:
    model.fit(training, output, n_epoch=200, batch_size=8, show_metric=True)
    model.save("model.tflearn")

model.fit(training, output, n_epoch=200, batch_size=8, show_metric=True)
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


# Erstellen der Datenbank als Mockup

import pandas as pd
import numpy as np

student_entries = np.array([
    ['7234562','Anna','Mueller','Marienstr.', '12','Duesseldorf','56789','1234','1.3','1245','2.3','5556',1],
    ['7166521','Markus','Shmidt','Volklingerstr.','5','Koeln','50667','5678','1.3','6547','2.0','4567',0],
    ['7345673','Maria','Xi','Feldweg','2','Duesseldorf','40210','1234','1,7','1245','2.0','4567',1],
    ['7623451','Philipp','Nowak','Galenstr','5','Dortmund','44137','5678','3.0','6547','1.0','5556',1 ],
    ['7122456','Christian','Klassen','Bachfeld','6','Dortmund','44138','1234','4.0','1245','1.3','5556',0]], dtype=object)
student_entries_df = pd.DataFrame(student_entries, columns = ['Matriculation_number','Name','Surname','road','house_number','city', 'postcode', 'Passed_Exam1', 'Passed_Exam1_Grade', 'Passed_Exam2','Passed_Exam2_Grade', 'Applied_Exam','SemesterFeePaid' ])
print(student_entries_df)

courses = np.array([
    ['1234', 'Physics'],
    ['5678', 'Economics'],
    ['1245', 'English'],
    ['6547', 'Mathematics1'],
    ['5556', 'Mathematics2'],
    ['4567', 'Object_Oriented_Programming'],
    ['1111', 'Parallel_Programming'],
    ['1112', 'Prolog_with_Applications'],
    ['1113', 'Compiler_Construction'],
    ['1114', 'Model_Driven_Software_Development']], dtype=object)
courses_df = pd.DataFrame(courses, columns = ['ID_Subject', 'Subjects_Name'])
print(courses_df)

# Wichtige Tags noch weiterer Usecases hinzufügen:
greeting = 'greeting'
goodbye = 'goodbye'
identification_tag = 'identification'
change_address_tag = 'change_address'
change_name = 'change_name'
exam_reg = 'exam_reg'
exam_dereg = 'exam_dereg'
paid = 'paid'
grade_examination_tag = 'grade_examination'
status_examination_registration_tag = 'status_examination_registration'

need_matriculation = {
    greeting: False,
    goodbye: False,
    identification_tag: False,
    change_address_tag: True,
    change_name: True,
    exam_reg: True,
    exam_dereg: True,
    paid: True,
    grade_examination_tag: True,
    status_examination_registration_tag: True
}

def chat():
    # Wichtige Zahlen vorderfinieren als 0
    matr_no = 0
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")

        bag = bag_of_words(inp, words)
        results = model.predict([bag])
        Logger.debug(1, 'Prediction results:', results)

        accuracy_map = {}
        for index, label in enumerate(labels):
            accuracy_map[label] = float(list(results)[0][index])

        Logger.debug(1, 'Accuracy Map:', accuracy_map)

        results_index = numpy.argmax(results)
        Logger.debug(1, 'Index of maximum value:', results_index)
        tag = labels[results_index]
        Logger.debug(1, 'Tag is:', tag)

        if tag == identification_tag:
            Logger.debug(1, 'Check for matriculation number in identification')
            match = match_matriculation_number_from_input(inp)
            if match and matriculation_number_exists(match.group()):
                matr_no = match.group()
                chatbot_out(f'Hey {get_full_name_by_matriculation_number(matr_no)}!')

            second_result_index = numpy.argsort(results)[0][-2]
            Logger.debug(1, 'Index of second largest value:', second_result_index)
            next_tag = labels[second_result_index]
            Logger.debug(1, 'Second tag:', next_tag)
            if accuracy_map[next_tag] >= 0.5:
                tag = next_tag
                Logger.debug(1, 'New tag is:', tag)
            else:
                Logger.debug(1, 'Accuracy for second tag too low:', accuracy_map[next_tag])
                chatbot_out('How can I help you?')
                continue

        # Wahrscheinlichkeit für den Tag ? print(model.score(results)) ab prozentzahl in tag

        Logger.debug(1, 'Need matriculation:', need_matriculation[tag])
        if need_matriculation[tag] and not matr_no:
            matr_no = check_for_matriculation_number(inp)

        # Input auf vorgeschriebene Muster/Nummern untersuchen und diese Speichern.
        # house_no, exam_no, citycode, matr_no = checkingNumbers(inp)

        # Greeting mit reinnehmen Intents
        if tag == greeting:
            responses = data["intents"][0]["responses"]
            chatbot_out(random.choice(responses))

        # Wann soll Chatbot ausgeschaltet werden?
        elif tag == goodbye:
            responses = data["intents"][1]["responses"]
            chatbot_out(random.choice(responses))
            break

        # Behandlung Use Case Umzug
        elif tag == change_address_tag:
            if unix:
                Logger.debug(1, 'Use-Case:', change_address_tag)
                Logger.debug(1, 'User Input:', inp)

                activated = get_activated_stems(bag, words)
                Logger.debug(1, 'Activated stems:', activated)

                inp = filter_input_by_stems(inp, activated)
                Logger.debug(1, 'Filtered User Input:', inp)

                processor = AddressProcessor()
                processor.process_address_input(inp)
                address = processor.address
                Logger.debug(1, 'Processed Address:', address.road, address.house_number, address.postcode, address.city)

                retries = 0
                if not retries and len(processor.empty_members) >= 4:
                    Logger.debug(1, 'Frist try and all members empty')
                    chatbot_out('Okay, what is your new address?')
                    inp = user_in()
                    inp = filter_input_by_stems(inp, activated)
                    Logger.debug(1, 'Filtered User Input:', inp)

                    processor.process_address_input(inp)
                    address = processor.address
                    Logger.debug(1, 'Processed Address:', address.road, address.house_number, address.postcode, address.city)

                while retries < 3 and len(processor.empty_members) > 0:
                    Logger.debug(1, 'Empty Members:', processor.empty_members)
                    Logger.debug(1, 'Gathering missing information try:', retries + 1)

                    empty_members = ", ".join(
                        list(map(AddressProcessor.address_member_labels.get, processor.empty_members)))
                    chatbot_out('Excuse me, I could not recognize the following parts of your address:', empty_members)
                    chatbot_out('Please enter the missing information separately.')

                    for member in processor.empty_members:
                        member_input = input(f'{AddressProcessor.address_member_labels[member]}: ')

                        if member == 'road':
                            address.road = member_input
                        if member == 'house_number':
                            address.house_number = member_input
                        if member == 'postcode':
                            address.postcode = member_input
                        if member == 'city':
                            address.city = member_input

                    processor.reprocess_address(address)
                    Logger.debug(1, 'Reprocessed Address:', processor.address.road, processor.address.house_number,
                          processor.address.postcode, processor.address.city)

                    retries += 1

                if retries >= 3 and processor.empty_members:
                    Logger.debug(1, 'Failed three times')
                    chatbot_out(
                        'I am having problems recognizing your address. Please try something different I can help you with.')
                else:
                    new_address = processor.address
                    change_address(matr_no, new_address)
                    chatbot_out(
                        f'Great, I changed your address to {new_address.road} {new_address.house_number} in {new_address.postcode} {new_address.city}')
            else:
                chatbot_out('Seems like you are using an inferior operating system.')
                chatbot_out('Please switch to an UNIX-based OS and be awesome.')

        elif tag == change_name:
            chatbot_out('Add your new surname.')
            d = user_in()
            student_entries_df["Surname"] = numpy.where(student_entries_df["Matriculation_number"] == matr_no, d, student_entries_df["Surname"])
            chatbot_out('Your surname has been successfully uploaded.')

        elif tag == grade_examination_tag:
            chatbot_out("Please enter the exam ID.")
            b = user_in()

            ppp = sqldf(f"SELECT ID_Subject={b} FROM courses_df WHERE ID_Subject={b}")
            p = sqldf(f"SELECT Passed_Exam1_Grade FROM student_entries_df WHERE Passed_Exam1={b} and Matriculation_number={matr_no}")
            pp = sqldf(f"SELECT Passed_Exam2_Grade FROM student_entries_df WHERE Passed_Exam2={b} and Matriculation_number={matr_no}")

            leer = p.empty
            lee = pp.empty
            le = ppp.empty
            aaab = leer
            aaa = lee
            aa = le

            if (aaab == 0) & (aaa != 0) & (aa == 0):
                chatbot_out("Your Grade for the entered Exam is", p.iat[0, 0])
            elif (aaab != 0) & (aaa == 0) & (aa == 0):
                chatbot_out("Your Grade for the entered Exam is", pp.iat[0, 0])
            elif (aaab != 0) & (aaa != 0) & (aa == 0):
                chatbot_out("I am sorry the given subject hasn't been passed yet.")
            else:
                chatbot_out("You entered a wrong number.")

        elif tag == status_examination_registration_tag:
            chatbot_out('Please enter the exam ID.')
            b = user_in()
            q = sqldf(f"SELECT Matriculation_number={matr_no}  FROM student_entries_df WHERE Passed_Exam1={b} and Matriculation_number={matr_no}")
            qq = sqldf(f"SELECT Matriculation_number={matr_no} FROM student_entries_df WHERE Passed_Exam2={b} and Matriculation_number={matr_no}")
            qqq = sqldf(f"SELECT Applied_Exam FROM student_entries_df WHERE Applied_Exam={b} and Matriculation_number={matr_no}")
            qqqq = sqldf(f"SELECT ID_Subject FROM courses_df WHERE ID_Subject={b}")

            isempty = q.empty
            isempt = qq.empty
            isemp = qqq.empty
            isem = qqqq.empty
            ter = isempty
            te = isempt
            tee = isemp
            teee = isem
            if (ter == 0) & (te != 0) & (tee != 0) & (teee == 0):
                chatbot_out("You already completed the Exam.")
            elif (ter != 0) & (te == 0) & (tee != 0) & (teee == 0):
                chatbot_out("You already completed the Exam.")
            elif (ter != 0) & (te != 0) & (tee == 0) & (teee == 0):
                chatbot_out("The exam was applied.")
            elif (ter != 0) & (te != 0) & (tee != 0) & (teee == 0):
                chatbot_out("You haven't registered for the exam yet.")
            else:
                chatbot_out('I have no information about the subject.')

        elif tag == exam_reg:
            exam_no = 0
            tries = 0
            while exam_no == 0 and tries < 3:
                tries += 1
                chatbot_out('Please enter your exam number.')
                inp = user_in()
                _, exam_no, _, _ = checkingNumbers(inp)
            if exam_no:
                register_exam(matr_no, exam_no)
            else:
                chatbot_out('I could not recognize the exam ID.')

        elif tag == exam_dereg:
            exam_no = 0
            tries = 0
            while exam_no == 0 and tries < 3:
                tries += 1
                chatbot_out('Please enter your exam number.')
                inp = user_in()
                _, exam_no, _, _ = checkingNumbers(inp)
            if exam_no:
                deregister_exam(matr_no, exam_no)
            else:
                chatbot_out('I could not recognize the exam ID.')

        elif tag == paid:
            if checkPaid(matr_no) == True:
                chatbot_out('You dont have any open payments.')
            else:
                chatbot_out('Your semester fee is not marked as paid right now.')

        # Restliche Fälle vernünftiges Handling für schrott eingaben etc finden
        else:
            chatbot_out('Sorry I did not understand that.')


# Hilfunsfunktionen
def checkingNumbers(inp):
    house_no, exam_no, citycode, matr_no = 0, 0, 0, 0
    inp_token = nltk.word_tokenize(inp)

    house_no_token = None

    for token in inp_token:
        # Checken ob Matrikelnummer, PLZ oder Prüfungsnummer in intents

        # Wenn Hausnummern dann ist token davor wsl -> Straße
        # Wenn PLZ ist token danach wsl -> Stadt
        try:
            number = int(token)
            if (len(token) == 4):
                exam_no = number
                print('exam') # TODO: Debugging
                print(exam_no)
            if (len(token) == 7):
                matr_no = number
                print('matr')
                print(matr_no)

        except:
            pass

    return house_no, exam_no, citycode, matr_no


def register_exam(matr_no, exam_no):
    matriculation_numbers = student_entries_df['Matriculation_number'].values
    subjects = courses_df['ID_Subject'].values

    if str(matr_no) in matriculation_numbers and str(exam_no) in subjects:
        student_row_index = numpy.where(matriculation_numbers == str(matr_no))[0]
        student_row = student_entries_df.iloc[student_row_index]
        registered_exam = student_row['Applied_Exam']

        if str(exam_no) == str(registered_exam) or exam_number_exists_for_student(exam_no, student_row):
            print('no update: subject_id already exists')

        else:
            print('update: register for exam')
            student_entries_df.loc[student_row_index, 'Applied_Exam'] = str(exam_no)

        print(student_entries_df)
    else:
        print('Invalid matriculation number or exam number, please check again')


def deregister_exam(matr_no , exam_no):
    matriculation_numbers = student_entries_df['Matriculation_number'].values
    applied_exams = student_entries_df['Applied_Exam'].values;

    if str(matr_no) in matriculation_numbers and str(exam_no) in applied_exams:
        student_row_index = numpy.where(matriculation_numbers == str(matr_no))[0]
        student_row = student_entries_df.iloc[student_row_index]
        registered_exam = student_row['Applied_Exam'][1]

        if str(exam_no) == str(registered_exam) and exam_number_exists_for_student(exam_no, student_row):
            print('update: deregister from exam')
            student_entries_df.loc[student_row_index, 'Applied_Exam'] = str(0)

        else:
            print('no update: you are not registered for the given exam_no: ' + str(exam_no))

        print(student_entries_df)
    else:
        print('Invalid matriculation number or exam number, please check again')


def exam_number_exists_for_student(exam_no, student_row):
    if str(exam_no) in student_row:
        return True
    else:
        return False



def change_address(matriculation_number, address):
    Logger.debug(1, 'Change Address for:', matriculation_number)
    Logger.debug(1, 'New Address:', address.road, address.house_number, address.postcode, address.city)

    index = student_entries_df.index[student_entries_df['Matriculation_number'] == str(matriculation_number)][0]
    Logger.debug(1, 'Index for given matriculation:', index)
    if index is not None:
        student_entries_df.loc[index, ['road', 'house_number', 'postcode', 'city']] = [address.road,
                                                                                       address.house_number,
                                                                                       address.postcode, address.city]
        Logger.debug(1, 'Data Row with new address:', student_entries_df.iloc[index])

    else:
        Logger.debug(1, 'No index found for:', matriculation_number)


def changeName(matr_no, name, surname):
    print(matr_no)


def checkPaid(matr_no):
    return bool(int(student_entries_df[student_entries_df.Matriculation_number==matr_no].SemesterFeePaid))


def get_activated_stems(bag_of_words, stems):
    activated = []
    for i in range(len(bag_of_words)):
        if bag_of_words[i] == 1:
            activated.append(stems[i])

    return activated


def filter_input_by_stems(input, stems):
    stems.append('in')  # Just in case ;-)
    for stem in stems:
        pattern = r'\b' + stem + r'.{0,3}?\s?\b'
        input = re.sub(pattern, '', input, flags=re.IGNORECASE)

    return input


chat()
