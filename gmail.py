from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import email
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
        num_requested_emails = 2000 
        email_count = 500
        email_list_response = service.users().messages().list(userId='me', maxResults=500, q="after:2022/02/01 before:2022/02/28").execute()
        email_list = email_list_response['messages']
        nextPageToken = None
        try:
          nextPageToken = email_list_response['nextPageToken']
        except:
          nextPageToken = None
          print('No next page token')
          pass
        while email_count < num_requested_emails and nextPageToken:
            email_list_response = service.users().messages().list(userId='me', maxResults=500, q="after:2022/02/01 before:2022/02/28", pageToken=nextPageToken).execute()
            # print(email_list_response['messages'])
            email_list.extend(email_list_response['messages'])
            try: 
              nextPageToken = email_list_response['nextPageToken']
            except:
              nextPageToken = None
              print('No next page token')
            email_count += 500
        results = []
        for email in email_list:
          if email['id']:
            results.append(service.users().messages().get(userId='me', id=email['id']).execute())
        print(f'{len(results)=}')
        metadatas = []
        docs = []
        skipped_emails = 0
        for item in results:
          for header in item['payload']['headers']:
            if header['name'] == 'Subject':
              subject = header['value']
            if header['name'] == 'From':
              email_from = header['value']
            if header['name'] == 'Received':
              received_date = header['value']
            if header['name'] == 'Date':
              date = header['value']

          if 'payload' in item and 'parts' in item['payload']:
            if 'data' in item['payload']['parts'][0]['body']:
              body = item['payload']['parts'][0]['body']['data']
            elif 'parts' in item['payload']['parts'][0]['parts'][0]:
              body = item['payload']['parts'][0]['parts'][0]['parts'][0]['body']['data']
            else: 
              body = item['payload']['parts'][0]['parts'][0]['body']['data']
          else:
            if 'data' in item['payload']['body']:
              body = item['payload']['body']['data']
            else:
              pass
          if not body or len(body) <20:
            skipped_emails +=1
          if len(body) > 20:
            body = base64.urlsafe_b64decode(body.encode('UTF-8'))
            try:
              body = body.decode('utf-8')
              splits = text_splitter(body)
              new_splits = []
              sources = []
              for i, split in enumerate(splits):
                # new_splits.append('SUBJECT: ', subject + '|' + 'EMAIL_FROM: ', email_from + '|' + 'RECEIVED DATE: ', json.dumps(received_date) + '|' + 'CONTENT: ', split)
                new_splits.append('SUBJECT: ' + subject + '|' + 'EMAIL_FROM: ' + email_from + '|' + 'RECEIVED DATE: ' + date + '|' + 'CONTENT: ' + split)
                sources.append({"source": subject+str(i)})
              docs.extend(new_splits)
              metadatas.extend(sources)
            except Exception as e:
              print(e)
              skipped_emails +=1
              pass

        print(f'{skipped_emails=}')
        print(f'{len(docs)=}')
        print(docs[0:10])
        for doc in docs:
          if len(doc) > 8000:
            print(doc[0:150])

        store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
        # This creates an index out of the store data structure
        if os.path.exists('faiss'):
          existing_index = FAISS.load_local('faiss', OpenAIEmbeddings())
          existing_index.add_texts(texts=docs, metadatas=metadatas)


          # existing_index = faiss.read_index("docs.index")
          # n = store.index.ntotal
          # d = store.index.d
          # print(store.index.reconstruct_n(0, n))
          # print(type(store.index.reconstruct_n))
          # print(len(store.index.reconstruct_n))
          # print(store.index.reconstruct_n.shape)
          # vectors = faiss.vector_to_array(store.index.reconstruct_n(0, n))
          # vectors = vectors.reshape(n, d)
          # existing_index.add(vectors)


          # Should probably stick to using FAISS (langchain) vs intermingling with faiss
          print(f'{os.getcwd()=}')
          existing_index.save_local('faiss')
          # faiss.write_index(existing_index, "docs.index")
        else:
          store.save_local('faiss')
          # faiss.write_index(store.index, "docs.index")
        # Now that it's saved to disk, we can remove the index from memory
        store.index = None
        # This keeps the vectors from the store accessible, but this is not related to the index
        # with open("faiss_store.pkl", "wb") as f:
        #     pickle.dump(store, f)

        # with open("text.pkl", "wb") as f:
        #     pickle.dump(docs, f)

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