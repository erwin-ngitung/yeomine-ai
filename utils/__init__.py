import re
import warnings
import json
import numpy as np
import shutil
import os
from zipfile import ZipFile

warnings.filterwarnings("ignore")


def label_name(num, maks):
    len_text = len(str(maks))
    len_num = len(str(num))
    name = "0" * (len_text - len_num)
    name += str(num)

    return name


def make_folder(path_file):
    directory1 = f'{path_file}/images'

    if not os.path.exists(directory1):
        os.makedirs(directory1)
    else:
        shutil.rmtree(directory1)
        os.makedirs(directory1)

    directory2 = f'{path_file}/videos'

    if not os.path.exists(directory2):
        os.makedirs(directory2)
    else:
        shutil.rmtree(directory2)
        os.makedirs(directory2)

    directory3 = f'{path_file}/annotations'
    if not os.path.exists(directory3):
        os.makedirs(directory3)
    else:
        shutil.rmtree(directory3)
        os.makedirs(directory3)


def make_folder_only(path_file):
    directory1 = f'{path_file}/images'

    if not os.path.exists(directory1):
        os.makedirs(directory1)

    directory2 = f'{path_file}/videos'

    if not os.path.exists(directory2):
        os.makedirs(directory2)

    directory3 = f'{path_file}/annotations'
    if not os.path.exists(directory3):
        os.makedirs(directory3)


def make_zip_only(path_folder, file_path, name):
    if os.path.exists(f'{path_folder}/{name}.zip'):
        os.remove(f'{path_folder}/{name}.zip')

    with ZipFile(f'{path_folder}/{name}.zip', 'w') as zip_object:
        zip_object.write(file_path, os.path.basename(file_path))


def make_zip(path_folder, name):
    if os.path.exists(f'{path_folder}/{name}.zip'):
        os.remove(f'{path_folder}/{name}.zip')

    # Create object of ZipFile
    with ZipFile(f'{path_folder}/{name}.zip', 'w') as zip_object:
        # Traverse all files in directory
        for folder_name, sub_folders, file_names in os.walk(f'{path_folder}/images'):
            for filename in file_names:
                # Create filepath of files in directory
                file_path = os.path.join(folder_name, filename)
                # Add files to zip file
                zip_object.write(file_path, os.path.basename(file_path))

        # Traverse all files in directory
        for folder_name, sub_folders, file_names in os.walk(f'{path_folder}/annotations'):
            for filename in file_names:
                # Create filepath of files in directory
                file_path = os.path.join(folder_name, filename)
                # Add files to zip file
                zip_object.write(file_path, os.path.basename(file_path))


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

