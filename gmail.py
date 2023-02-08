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
        email_list_response = service.users().messages().list(userId='me', labelIds=['UNREAD'], maxResults=2).execute()
        results = []
        email_list = email_list_response['messages']
        print(email_list)
        for email in email_list:
          if email['id']:
            results.append(service.users().messages().get(userId='me', id=email['id']).execute())
        for item in results:
          delivered_to = item['payload']['headers'][0]
          print(type(delivered_to))
          received_date = item['payload']['headers'][1]
          body = item['payload']['parts'][0]['body']['data']
          # Decode the base64 encoded data
          body = base64.urlsafe_b64decode(body.encode('ASCII'))
          # Convert the decoded data to a string
          body = body.decode('utf-8')
          text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
          # docs = [body]
          # metadatas = []
          # for i, d in enumerate(data):
          #     splits = text_splitter.split_text(d)
          #     docs.extend(splits)
          #     metadatas.extend([{"source": sources[i]}] * len(splits))


          # Here we create a vector store from the documents and save it to disk.
          delivered_str = json.dumps(delivered_to)
          received_str = json.dumps(received_date)
          metadata = delivered_str + received_str
          print(body)
          print(metadata)

          store = FAISS.from_texts(body, OpenAIEmbeddings(), metadatas=metadata)
          # This creates an index out of the store data structure
          faiss.write_index(store.index, "docs.index")
          # Now that it's saved to disk, we can remove the index from memory
          store.index = None
          # This keeps the vectors from the store accessible, but this is not related to the index
          with open("faiss_store.pkl", "wb") as f:
              pickle.dump(store, f)



    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    main()