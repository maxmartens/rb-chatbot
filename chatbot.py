from sys import platform

unix = "darwin" in platform or "linux" in platform
debug_mode = True

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

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


def debug(*vars):
    if debug_mode:
        print('[DEBUG]', vars)


def match_matriculation_number_from_input(input):
    debug('Search for matriculation number in input')
    regex = r'(?<!\d)\d{7}(?!\d)'
    debug('Searching RegEx in input:', regex, input)
    return re.search(regex, input)


def matriculation_number_exists(matriculation_number):
    debug('Check if matriculation number exists:', matriculation_number)
    candidate_indices = student_entries_df.index[student_entries_df['Matriculation_number'] == str(matriculation_number)]
    debug('Indices for given matriculation number:', list(candidate_indices))
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
    with open("model.tflearn") as chkpt:
        model.load(chkpt)
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


# Erstellen der Datenbank als Mockup

import pandas as pd
import numpy as np

student_entries = np.array([['7234562', 'Anna', 'Mueller', 'Marienstr.', '12', 'Duesseldorf', '56789', '1234', '1.3', '1245', '2.3', '5556'], ['7166521', 'Markus', 'Shmidt', 'Volklingerstr.', '5', 'Koeln', '50667', '5678', '1.3', '6547', '2.0', '4567'], ['7345673', 'Maria', 'Xi', 'Feldweg', '2', 'Duesseldorf', '40210', '1234', '1.7', '1245', '2.0', '4567'], ['7623451', 'Philipp', 'Nowak', 'Galenstr', '5', 'Dortmund', '44137', '5678', '3.0', '6547', '1.0', '5556'], ['7122456', 'Christian', 'Klassen', 'Bachfeld', '6', 'Dortmund', '44138', '1234', '4.0', '1245', '1.3', '5556']], dtype=object)
student_entries_df = pd.DataFrame(student_entries, columns = ['Matriculation_number', 'Name', 'Surname', 'road', 'house_number', 'city', 'postcode', 'Passed_Exam1', 'Passed_Exam1_Grade', 'Passed_Exam2', 'Passed_Exam2_Grade', 'Applied_Exam'])
print(student_entries_df)

courses = np.array([['1234', 'Physics'], ['5678', 'Economics'], ['1245', 'English'], ['6547', 'Mathematics1'], ['5556', 'Mathematics2'], ['4567', 'Object_Oriented_Programming'], ['1111', 'Parallel_Programming'], ['1112', 'Prolog_with_Applications'], ['1113', 'Compiler_Construction'], ['1114', 'Model_Driven_Software_Development']], dtype=object)
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

need_matriculation = {
    greeting: False,
    goodbye: False,
    identification_tag: False,
    change_address_tag: True,
    change_name: True,
    exam_reg: True,
    exam_dereg: True,
    paid: True
}

def chat():
    # Wichtige Zahlen vorderfinieren als 0
    matr_no = 0
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")

        bag = bag_of_words(inp, words)
        results = model.predict([bag])
        debug('Prediction results:', results)

        accuracy_map = {}
        for index, label in enumerate(labels):
            accuracy_map[label] = float(list(results)[0][index])

        debug('Accuracy Map:', accuracy_map)

        results_index = numpy.argmax(results)
        debug('Index of maximum value:', results_index)
        tag = labels[results_index]
        debug('Tag is:', tag)

        if tag == identification_tag:
            second_result_index = numpy.argsort(results)[0][-2]
            debug('Index of second largest value:', second_result_index)
            tag = labels[second_result_index]
            debug('New tag is:', tag)

        # Wahrscheinlichkeit für den Tag ? print(model.score(results)) ab prozentzahl in tag

        debug('Need matriculation:', need_matriculation[tag])
        if need_matriculation[tag] and not matr_no:
            matr_no = check_for_matriculation_number(inp)

        # Input auf vorgeschriebene Muster/Nummern untersuchen und diese Speichern.
        # house_no, exam_no, citycode, matr_no = checkingNumbers(inp)

        # Greeting mit reinnehmen Intents
        if tag == greeting:
            responses = data["intents"][0]["responses"]
            print(chatbotname + random.choice(responses))

        # Wann soll Chatbot ausgeschaltet werden?
        elif tag == goodbye:
            responses = data["intents"][1]["responses"]
            print(chatbotname + random.choice(responses))
            break

        # Behandlung Use Case Umzug
        elif tag == change_address_tag:
            if unix:
                debug('Use-Case:', change_address_tag)
                debug('User Input:', inp)

                activated = get_activated_stems(bag, words)
                debug('Activated stems:', activated)

                inp = filter_input_by_stems(inp, activated)
                debug('Filtered User Input:', inp)

                processor = AddressProcessor()
                processor.process_address_input(inp)
                address = processor.address
                debug(f'Processed Address:', address.road, address.house_number, address.postcode, address.city)

                retries = 0
                while retries < 3 and len(processor.empty_members) > 0:
                    debug('Empty Members:', processor.empty_members)
                    debug('Gathering missing information try:', retries + 1)

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
                    debug('Reprocessed Address:', processor.address.road, processor.address.house_number,
                          processor.address.postcode, processor.address.city)

                    retries += 1

                if retries >= 3 and processor.empty_members:
                    debug('Failed three times')
                    chatbot_out('I am having problems recognizing your address. Please try something different I can help you with.')
                else:
                    new_address = processor.address
                    change_address(matr_no, new_address)
                    chatbot_out(
                        f'Great, I changed your address to {new_address.road} {new_address.house_number} in {new_address.postcode} {new_address.city}')

            else:
                chatbot_out('Seems like you are using an inferior operating system.')
                chatbot_out('Please switch to an UNIX-based OS and be awesome.')

        elif tag == change_name:
            print(change_name)

        elif tag == exam_reg:
            while (matr_no == 0):  # Solange User keine Matrikelnummer eingibt hier gefangen
                print(chatbotname + 'Please enter your matriculation number.')
                inp = input('You:')
                _, _, _, matr_no = checkingNumbers(inp)

            exam_no = 0
            while (exam_no == 0):
                print(chatbotname + 'Please enter your exam number.')
                inp = input('You:')
                _, exam_no, _, _ = checkingNumbers(inp)
            register_exam(matr_no, exam_no)
            print(student_entries_df)


        elif tag == exam_dereg:
            while matr_no or exam_no:  # Solange User keine Matrikelnummer eingibt hier gefangen
                print(chatbotname + 'Please enter your matriculation number.')
                inp = input('You:')
                _, _, _, matr_no = checkingNumbers(inp)

            exam_no = 0
            while (exam_no == 0):
                print(chatbotname + 'Please enter your exam number.')
                inp = input('You:')
                _, exam_no, _, _ = checkingNumbers(inp)
            deregister_exam(matr_no, exam_no)

        elif tag == paid:
            if checkPaid(matr_no) == True:
                print(chatbotname + 'You dont have any open payments')
            else:
                print(chatbotname + 'Your semesterfee is not marked as paid right now.')

        # Restliche Fälle vernünftiges Handling für schrott eingaben etc finden
        else:
            print(chatbotname + 'Sorry i didnt understand that.')


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
            if (000 <= number <= 999):
                house_no = number
                print('housenumber')
                print(house_no)
                print(inp_token)
            if (1000 <= number <= 9999):
                exam_no = number
                print('exam')
                print(exam_no)
            if (10000 <= number <= 99999):
                citycode = number
                print('city')
                print(citycode)
            if (1000000 <= number <= 9999999):
                matr_no = number
                print('matr')
                print(matr_no)
        except:
            pass
    return house_no, exam_no, citycode, matr_no


def register_exam(matr_no, exam_no):
    # check for matr_no and exam_no
    matriculation_numbers = student_entries_df['Matriculation_number'].values

    if str(matr_no) in matriculation_numbers and str(exam_no) in courses_df['ID_Subject'].values:
        student_row_index = numpy.where(matriculation_numbers == str(matr_no))[0]
        student_row = student_entries_df.iloc[student_row_index]
        registered_exam = student_row['Applied_Exam']

        if str(exam_no) == str(registered_exam):
            print('no update: subject_id already exists')

        else:
            print('update: register for exam')
            student_entries_df.loc[student_row_index, 'Applied_Exam'] = str(exam_no)

        print(student_entries_df)
    else:
        print('Invalid matriculation number or exam number, please check again')


def deregister_exam(matr_no , exam_no):
    # check for matr_no and exam_no
    matriculation_numbers = student_entries_df['Matriculation_number'].values

    if str(matr_no) in matriculation_numbers and str(exam_no) in student_entries_df['Applied_Exam'].values:
        student_row_index = numpy.where(matriculation_numbers == str(matr_no))[0]
        student_row = student_entries_df.iloc[student_row_index]
        registered_exam = student_row['Applied_Exam'][1]

        if str(exam_no) == str(registered_exam):
            print('update: deregister from exam')
            student_entries_df.loc[student_row_index, 'Applied_Exam'] = str(0)

        else:
            print('no update: you are not registered for the given exam_no: ' + str(exam_no))

        print(student_entries_df)
    else:
        print('Invalid matriculation number or exam number, please check again')


def change_address(matriculation_number, address):
    debug('Change Address for:', matriculation_number)
    debug('New Address:', address.road, address.house_number, address.postcode, address.city)

    index = student_entries_df.index[student_entries_df['Matriculation_number'] == str(matriculation_number)][0]
    debug('Index for given matriculation:', index)
    if index is not None:
        student_entries_df.loc[index, ['road', 'house_number', 'postcode', 'city']] = [address.road, address.house_number, address.postcode, address.city]
        debug('Data Row with new address:', student_entries_df.iloc[index])

    else:
        debug('No index found for:', matriculation_number)


def changeName(matr_no, name, surname):
    print(matr_no)


def checkPaid(matr_no):
    return True


def get_activated_stems(bag_of_words, stems):
    activated = []
    for i in range(len(bag_of_words)):
        if bag_of_words[i] == 1:
            activated.append(stems[i])

    return activated

def filter_input_by_stems(input, stems):
    for stem in stems:
        pattern = r'\b' + stem + r'.{0,3}?\s?\b'
        input = re.sub(pattern, '', input, flags=re.IGNORECASE)

    return input

# Matr No noch auslagern? -> nicht null und valide !

# Matr No noch auslagern? -> nicht null und valide !


chat()