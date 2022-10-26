import nltk
import os
import math


train_data =os.path.join("data1/train/")
test_data =os.path.join("data1/dev/")

ham = "ham/"
spam = "spam/"
MAX_NUM = 5000 
K = 2

# Sort the dictionary by value in descending order 
def sort_by_value(d):
    sortedd=sorted(d.items(),key=lambda a:a[1], reverse = True)
    return sortedd
    
#Process words, delete whitespace, and convert all uppercase to lowercase
def word_process(word):
    word_low = word.strip().lower()
    word_final = word_low
    return word_final

#add to dictionary
def add_to_dict(word, dict_name):
    if(word in dict_name):
        num = dict_name[word]
        num += 1
        dict_name[word] = num
    else:
        dict_name[word] = 1

def negative_dict_maker(dictionary):
    d = dict()
    for (key, value) in dictionary.items():
        if(value >= MAX_NUM or value <= 1):       
            d[key]=1
    return d

#txt file reader
def text_reader(file_name, dict_name):
    tokenizer = nltk.RegexpTokenizer("[\w']{3,}")   
    f = open(file_name, 'r',  encoding = "ISO-8859-1")
    for line in f:
        words = tokenizer.tokenize(line)
        for word in words:
            word = word_process(word)
            add_to_dict(word, dict_name) #add to dictionary 
    f.close()

def save_dict(dict_name, file_path, all_flag):
    f = open(file_path, 'w',  encoding = "ISO-8859-1")
    word_max = "" #Record the word with the most occurrences
    value_max = 0 #Record the number of occurrences of the most occurring word
    for (key, value) in dict_name.items():
        if(not all_flag):
            if value > 1 and value < MAX_NUM:
                f.writelines(key+" "+str(value)+"\n")
            if value > value_max:
                word_max = key
                value_max = value
        else:
            f.writelines(key+" "+str(value)+"\n")
            if value > value_max:
                word_max = key
                value_max = value
    f.close()
    print("Hamgiin ih garj irsen ug: "+word_max+", Count: "+str(value_max))
        
def load_dict(file_path):
    dict_loaded = dict()
    f = open(file_path, 'r', encoding = "ISO-8859-1")
    
    while 1:
        line = f.readline()
        if not line:
            break
        words = line.split()
        dict_loaded[words[0]] = int(words[1])
    f.close()
    return dict_loaded

# number of normal emails, 
# the number of spam emails, 
# and the total number in a data file
def save_file_number(ham, spam, total):
    f = open("file_number.data", 'w',  encoding = "ISO-8859-1")
    f.writelines(str(ham)+"\n")
    f.writelines(str(spam)+"\n")
    f.writelines(str(total)+"\n")
    f.close()

#create dictionary and calculate the number of ham or spam
def traverse_dictionary_maker(file_path):
    dictionary = dict()
    ham_path = file_path+ham
    spam_path = file_path+spam
    path = {ham_path,spam_path}
    path_order = 0
    num_ham = 0
    num_spam = 0
    for i in path:#From ham folder to spam folder
        folders = os.listdir(i)   #Get the names of all files in a folder
        for file_name in folders:
            if os.path.isfile(i+file_name):
                text_reader(i+file_name, dictionary)  #Add words from mail to dictionary
                if(path_order == 0):
                    num_ham += 1
                else:
                    num_spam += 1
        path_order += 1

    save_file_number(num_ham, num_spam, num_ham + num_spam)#training
    return dictionary

#spam/normal mail dictionary
def dict_creator(file_path,negative_dict):
    dictionary = load_dict("dict_file.data")
    for key in dictionary:#Record the number of occurrences of each word in the total dictionary as 0
        dictionary[key] = 0
    if(not os.path.isfile(file_path)): 
        folders = os.listdir(file_path)#Extract the filenames of all files in the folder
        for file_name in folders:
            raw_dict = dict()
            if os.path.isfile(file_path+file_name):
                text_reader(file_path+file_name, raw_dict)
            for key in raw_dict: #words that appear in the normal dictionary but not in the spam dictionary         
                if key not in negative_dict:
                    num = dictionary[key]
                    num += 1
                    dictionary[key] = num
    else:
        raw_dict = dict()
        text_reader(file_path, raw_dict)
        for key in raw_dict: #words that appear in the normal dictionary but not in the spam word dictionary             
            if key not in negative_dict:
                raw_dict[key]=1
        dictionary=raw_dict

    return dictionary
    

#Read the previously stored train folder (number of normal emails, number of spam emails, total)
def read_w_number():
    f = open("file_number.data", 'r',  encoding = "ISO-8859-1")
    lines = f.readlines()
    w_num = [int(lines[0]), int(lines[1]), int(lines[2])]
    f.close()
    return w_num

#Output the first 30 words of the dictionary
def print_top(list_name):
    index = 0
    while(index < 30):
        print(list_name[index])
        index += 1

def calculate(word, dict_name, n_w, exist_flag):
    if(exist_flag):
        result = math.log(dict_name[word]+1)
    else:
        result = math.log(n_w+K-dict_name[word])
    return result

def bayes_score(vector, dict_name, n_w):
    result = 0.0
    for (key, value) in dict_name.items():
       exist_flag = (key in vector)
       result += calculate(key, dict_name, n_w, exist_flag)
       result -= math.log(n_w)
    return result

def predict(file_path, w_num, ham_dict, spam_dict):
    vector = dict_creator(file_path,negative_dict)
    prob_ham = bayes_score(vector, ham_dict, w_num[0])
    prob_spam = bayes_score(vector, spam_dict, w_num[1])
    if prob_ham > 1.15*prob_spam:
        return 0
    else :
        return 1
    

file_path = train_data
dictionary = traverse_dictionary_maker(file_path)
negative_dict = negative_dict_maker(dictionary) 

save_dict(dictionary, "dict_file.data", False) 
dictionary = load_dict("dict_file.data")
w_num = read_w_number()

#==================================================================================#

print("-------------------------------------------")
print("Train folder info: ")
print("Jiriin mail: "+str(w_num[0])+"\nSpamd tootsogdson mail: "+str(w_num[1])+"\nNiit mail: "+str(w_num[2]))

print("HAM")
ham_dict = dict_creator(train_data + ham, negative_dict)
save_dict(ham_dict, "ham_dict.data", True)
print("Ham dictionary amjilttai.")

print("SPAM")
spam_dict = dict_creator(train_data + spam, negative_dict)
save_dict(spam_dict, "spam_dict.data", True)
print("Spam dictionary amjilttai.")
print("-------------------------------------------")

print("Ham message deerh top 30n ug: ")
list_ham = sort_by_value(ham_dict)
print_top(list_ham)
print("-------------------------------------------")
print("Spam message deerh top 30n ug: ")
list_spam = sort_by_value(spam_dict)
print_top(list_spam)

def test(data_set):
    test_set = [ham, spam]
    correctrate=0
    testnum=0
    for w in test_set:
        correctnum=0
        folder_path = data_set + w 
        print (folder_path)
        files = os.listdir(folder_path)
        total_num = 0
        ham_predict_num = 0
        spam_predict_num = 0
        print(len(files))
        f = open("file.data", 'w',  encoding = "ISO-8859-1")
        f.writelines(folder_path+"\n")  
        for file in files:#
            if os.path.isfile(folder_path+file):
                f.writelines(folder_path+file+"\n")
                result = predict(folder_path+file, w_num, ham_dict, spam_dict)  
            
                total_num += 1
                if(result == 0):
                    ham_predict_num += 1
                    if(w == ham):
                        correctnum+=1                  
                else:
                    spam_predict_num += 1
                    if (w == spam):
                        correctnum+=1
                        
        correctrate+=correctnum
        testnum+=total_num
        f.close()    
        print("Niit mail: " + str(total_num))
        print("Normal mail: " + str(ham_predict_num))
        print("Spamd tootsogdson mail: " + str(spam_predict_num))
        print("Accuracy: "+str(correctnum/total_num))
        print("-------------------------------------------")
    print("Correct rate："+str(correctrate/testnum))

print("-------------------------------------------")
print("Dev folder deerh test")
test(test_data)



print("------------------exiting------------------")




