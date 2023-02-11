from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# import email
import base64
import json
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from bs4 import BeautifulSoup
import re


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
        email_list_response = service.users().messages().list(userId='me', maxResults=1000, q="after:2022/02/01 before:2022/02/28").execute()
        results = []
        email_list = email_list_response['messages']
        # print(email_list)
        for email in email_list:
          if email['id']:
            results.append(service.users().messages().get(userId='me', id=email['id']).execute())
        print(f'{len(results)=}')
        metadatas = []
        docs = []
        for item in results:
          for header in item['payload']['headers']:
            if header['name'] == 'Subject':
              subject = header['value']
            if header['name'] == 'From':
              email_from = header['value']
            if header['name'] == 'Received':
              received_date = header['value']
              # print(f'{received_date=}')
          # print(item['payload'])

          if 'payload' in item and 'parts' in item['payload']:
            if 'data' in item['payload']['parts'][0]['body']:
              body = item['payload']['parts'][0]['body']['data']
            elif 'parts' in item['payload']['parts'][0]['parts'][0]:
              # print(item['payload']['parts'])
              # print('here')
              # print(item['payload']['parts'][0]['parts'])
              # print('here')
              # print(item['payload']['parts'][0]['parts'][0]['parts'])
              # print('here')
              body = item['payload']['parts'][0]['parts'][0]['parts'][0]['body']['data']
            else: 
              body = item['payload']['parts'][0]['parts'][0]['body']['data']
          else:
            body = item['payload']['body']['data']

          if len(body) > 20:
            # Decode the base64 encoded data
            body = base64.urlsafe_b64decode(body.encode('UTF-8'))
            # Convert the decoded data to a string
            body = body.decode('utf-8')
            # body = body.replace("-","+").replace("_","/")
            # decoded_data = base64.b64decode(body)
            # print(f'{decoded_data=}')
            # '; |, |\*|\n
            # text_splitter = CharacterTextSplitter(chunk_size=3900, separator=". | , | \n | ; | :")
            # splits = text_splitter.split_text(body)
            splits = text_splitter(body)
            # print(f'{splits=}')
            new_splits = []
            sources = []
            for i, split in enumerate(splits):
              new_splits.append(subject + '|' + email_from + '|' + json.dumps(received_date) + '|' + split)
              sources.append({"source": subject+str(i)})
            docs.extend(new_splits)
            metadatas.extend(sources)
        print(f'{len(docs)=}')
        for doc in docs:
          # print(f'{doc=}')
          if len(doc) > 8000:
            print(doc[0:150])
        # print(f'{docs[1]=}')
        # print(f'{docs[2]=}')
        # print(f'{docs[3]=}')
        # print(f'{docs[4]=}')
        # print(f'{docs[5]=}')
        # print(f'{docs[6]=}')
        # print(f'{docs[7]=}')
        # print(f'{docs[8]=}')
        # print(f'{docs[9]=}')
        # print(f'{docs[10]=}')                                                        
        # print('\n'.join(map(str, docs)))
        # print(docs)
        # print(metadatas)

          # store = FAISS.from_texts(docs, OpenAIEmbeddings())
        # Might need to iterate through docs and filter out any with token length >8000

        store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
        # This creates an index out of the store data structure
        faiss.write_index(store.index, "docs.index")
        # Now that it's saved to disk, we can remove the index from memory
        store.index = None
        # This keeps the vectors from the store accessible, but this is not related to the index
        with open("faiss_store.pkl", "wb") as f:
            pickle.dump(store, f)
        print(f'{len(docs)=}')
        with open("text.pkl", "wb") as f:
            pickle.dump(docs, f)

        # with open("text.pkl", "rb") as f:
        #   text_list = pickle.load(f)
          # print(f'{len(text_list)=}')
          # print(f'{text_list[2]=}')


    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')


def text_splitter(text):
    # print(text)
    chunks = []
    chunk = ""
    # naive_split = text.split(". | , | \n | ; | :")
    text = re.sub('\r', '', text)
    naive_split = re.split(r"[,;:\n]", text)
    # print(naive_split)
    # print(len(naive_split))
    for split in naive_split:
      # print(split)
      if len(chunk) < 3900:
        # print('here')
        chunk += split
        # print(chunk)
      else:
        # print(chunk)
        chunks.append(chunk)
        chunk = ""
    chunks.append(chunk)
    # print(chunks)
    return chunks

if __name__ == '__main__':
    main()