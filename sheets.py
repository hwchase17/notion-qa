from __future__ import print_function

import os.path
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError



class Edit_Sheet:

  def __init__(self):
    # If modifying these scopes, delete the file token.json.
    self.SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    # The ID and range of a sample spreadsheet.
    self.SAMPLE_SPREADSHEET_ID = '1MPYQWe3b8MfBdigk13cEuxdwXuzt-_6qEgHux1fwND4'
    self.SAMPLE_RANGE_NAME = 'B1:B3'

  def update(self, data):
      """Shows basic usage of the Sheets API.
      Prints values from a sample spreadsheet.
      """
      creds = None
      # The file token.json stores the user's access and refresh tokens, and is
      # created automatically when the authorization flow completes for the first
      # time.
      if os.path.exists('token.json'):
          creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
      # If there are no (valid) credentials available, let the user log in.
      if not creds or not creds.valid:
          if creds and creds.expired and creds.refresh_token:
              creds.refresh(Request())
          else:
              flow = InstalledAppFlow.from_client_secrets_file(
                  'credentials.json', self.SCOPES)
              creds = flow.run_local_server(port=0)
          # Save the credentials for the next run
          with open('token.json', 'w') as token:
              token.write(creds.to_json())

      try:
          service = build('sheets', 'v4', credentials=creds)

          # data = {
          #  "value": 5
          # }         

          # json_data = json.dumps(data)
          # data = [[5]]

          # Call the Sheets API
          sheet = service.spreadsheets()
          response = sheet.values().update(spreadsheetId=self.SAMPLE_SPREADSHEET_ID,
                                      range=self.SAMPLE_RANGE_NAME, valueInputOption="USER_ENTERED", body={"values": data}).execute()
          # values = result.get('values', [])

          # if not values:
          #     print('No data found.')
          #     return
          
          # print(values)
          # print('Name, Major:')
          # for row in values:
          #     # Print columns A and E, which correspond to indices 0 and 4.
          #     print('%s, %s' % (row[0], row[4]))
      except HttpError as err:
          print(err)


# if __name__ == '__main__':
#     main()