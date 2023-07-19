import re
import warnings
import json
import numpy as np
import shutil 
import os.path
import subprocess

warnings.filterwarnings("ignore")


def label_name(num, maks):
    len_text = len(str(maks))
    len_num = len(str(num))
    name = "0" * (len_text - len_num)
    name += str(num)

    return name


def make_zip(weight_name):
    # Creating the ZIP file 
    archived = shutil.make_archive(f'weights/{weight_name}.pth', 'zip', f'weights/{weight_name}.zip')

    if os.path.exists(f'weights/{weight_name}.zip'):
        print(archived) 
    else: 
        print("ZIP file not created")


def update_json(name, username, email, password):
    data = open('data/data_account.json')

    data_account = json.load(data)

    name = data_account['name'] + [name]
    username = data_account['username'] + [username]
    email = data_account['email'] + [email]
    password = data_account['password'] + [password]

    data.close()

    data_email = {'name': name,
                  'username': username,
                  'email': email,
                  'password': password}

    with open('data/data_account.json', 'w') as json_file:
        json.dump(data_email, json_file)

    return None


def replace_json(name, username, old_email, new_email, password):
    data = open('data/data_account.json')

    data_account = json.load(data)

    index = np.where(np.array(data_account['email']) == old_email)[0][0]
    data_account['name'][index] = name
    data_account['username'][index] = username
    data_account['email'][index] = new_email
    data_account['password'][index] = password

    data.close()

    data_email = {'name': data_account['name'],
                  'username': data_account['username'],
                  'email': data_account['email'],
                  'password': data_account['password']}

    with open('data/data_account.json', 'w') as json_file:
        json.dump(data_email, json_file)

    return None


def check_account(name_email, name_password):
    data = open('data/data_account.json')

    data_email = json.load(data)

    name = data_email['name']
    username = data_email['username']
    email = data_email['email']
    password = data_email['password']

    index = np.where(np.array(email) == name_email)[0][0]
    password_true = password[index]

    if name_email in email and name_password == password_true:
        return name[index], username[index], 'register'
    if name_email in email and name_password != password_true:
        return '', '', 'wrong password'
    if name_email not in email:
        return '', '', 'not register'


def check_email(email):
    data = open('data/data_account.json')

    data_email = json.load(data)

    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    if re.fullmatch(regex, email):
        if email not in data_email['email']:
            value = "valid email"
        else:
            value = "duplicate email"
    else:
        value = "invalid email"

    return value

