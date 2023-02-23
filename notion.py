
from notion_client import APIResponseError
import notion_client
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import numpy as np
from datetime import datetime
import time
import re
import concurrent.futures
import threading
import os

"""
This version takes Jan 2022-March 2022 inclusive in CRM + a small fake DB + a few docs. 
An entire DB row is stuffed into an embedding
"""

"""
TODO:
1. Add support for more block types
2. Clean up how pages are split
3. DONE Add metadata sources
4. Clean up database row entries / play around with formatting that helps OpenAI the most
5. DONE(for CRM pages) Consider adding parent page to the entry or metadata
6. FIXED I think page contents on the CRM entry aren't getting pulled properly - check that
7. The drop down toggles on this page aren't being processed. Everything else is though https://www.notion.so/moonchaser/Pre-offer-148bba51d4b648b5b67661e68ee16d98


Database name: TestDB, Created Time: 2023-02-22T15:01:00.000Z, Last Edited Time: 2023-02-22T15:02:00.000Z
column-1 name: Details
column-1 value: Bain consultant and then PM at Facebook and then co-founder at Moonchaser 
column-1 data type: rich_text
column-2 name: Tags 
column-2 value: Driven
column-2 data type: multi_select
column-3 name: Nam
column-3 value: David Patterson-Cole
column-3 data type: title
"""


moonchaser_key = os.environ.get('MOONCHASER_NOTION_KEY') 
print(moonchaser_key)
notion = notion_client.Client(auth=moonchaser_key)

# extract content from block based on type 
parsed_ids = {}
block_ids = {}
notion_call_count = 0
parsed_ids_lock = threading.Lock()
block_ids_lock = threading.Lock()
notion_calls_lock = threading.Lock()

def add_id(page_id):
    with parsed_ids_lock:
        if page_id in parsed_ids:
            print("This page has already been parsed. Error")
        print("adding page id: " + page_id + " to parsed_ids at time " + str(datetime.now()))
        parsed_ids[page_id] = True

def add_block_id(block_id):
    with block_ids_lock:
        if block_id in block_ids:
            print("This block has already been parsed. Error")
        print("adding block id: " + block_id + " to block_ids at time " + str(datetime.now()))
        block_ids[block_id] = True

def execute_notion_with_retry(func, *args, **kwargs):
    global notion_call_count
    with notion_calls_lock:
        notion_call_count = notion_call_count + 1
    for i in range(10):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # check if error contains string "rate limited"
            print('an unknown error occurred, message is: %s' % str(e))
            if "rate limited" in str(e):
                print('rate limited, waiting 10 seconds * i: ', i)
                time.sleep(30*i)
            else:
                time.sleep(1)
    return None


# def get_page_source(page_id, notion):
#     page = execute_notion_with_retry(notion.pages.retrieve, page_id)
#     if page['properties'] is not None and 'title' in page['properties'] and len(page['properties']['title']['title']) > 0:
#         return page['properties']['title']['title'][0]['plain_text']
#     return ""


def get_db_source(database_id, notion):
    db = execute_notion_with_retry(notion.databases.retrieve, database_id)
    if db['title'] is not None and len(db['title']) > 0:
        return db['title'][0]['plain_text']
    return "Source unknonwn for database " + database_id


def extract_text(rich_text):
    content = ""
    for entry in rich_text:
        content += entry['plain_text']
        if entry['href'] is not None:
            content += " url (" + entry['href'] + ")"
    return content

def extract_block_info(block):
    if block['id'] is not None:
        add_block_id(block['id'])
        
    content = None
    if block['type'] == 'paragraph':
        if block['paragraph']['rich_text']:
            content = extract_text(block['paragraph']['rich_text'])
    elif block['type'] == 'heading_1':
        if block['heading_1']['rich_text']:
            content = extract_text(block['heading_1']['rich_text'])
    elif block['type'] == 'heading_2':
        if block['heading_2']['rich_text']:
            content = extract_text(block['heading_2']['rich_text'])
    elif block['type'] == 'heading_3':
        if block['heading_3']['rich_text']:
            content = extract_text(block['heading_3']['rich_text'])
    elif block['type'] == 'bulleted_list_item':
        if block['bulleted_list_item']['rich_text']:
            content = extract_text(block['bulleted_list_item']['rich_text'])
    elif block['type'] == 'numbered_list_item':
        if block['numbered_list_item']['rich_text']:
            content = extract_text(block['numbered_list_item']['rich_text'])
    elif block['type'] == 'to_do':
        if block['to_do']['rich_text']:
            content = extract_text(block['to_do']['rich_text'])
    elif block['type'] == 'toggle':
        if block['toggle']['rich_text']:
            content = extract_text(block['toggle']['rich_text'])
    elif block['type'] == 'quote':
        if block['quote']['rich_text']:
            content = extract_text(block['quote']['rich_text'])
    elif block['type'] == 'code':
        if block['code']['title']:
            content = block['code']['title'][0]['plain_text']
    elif block['type'] == 'embed':
        if block['embed']['caption']:
            content = block['embed']['caption'][0]['plain_text']
    elif block['type'] == 'image':
        if block['image']['caption']:
            content = block['image']['caption'][0]['plain_text']
    elif block['type'] == 'video':
        if block['video']['caption']:
            content = block['video']['caption'][0]['plain_text']
    elif block['type'] == 'child_page':
        if block['child_page']['title']:
            content = block['child_page']['title']
    elif block['type'] == 'template':
         if block['template']['rich_text']:
            content = extract_text(block['template']['rich_text'])
    else:
        # print(block['type'])
        content = None
    
    return content

def block_parser(block: dict, notion: "notion_client.client.Client")-> dict:
    #if the block will already be parsed later on then we only want to add the title here
    content = extract_block_info(block)
    if block["type"] in ['page', 'child_page', 'child_database']:
        return content
    
    if content == None:
        # print("returning None  for block: type: ", block["type"])
        return None
        
    if block["has_children"]:
        block["children"] = []
        start_cursor = None
        while True:
            if start_cursor is None:
                blocks = execute_notion_with_retry(notion.blocks.children.list, block["id"])
            if blocks == None:
                break
            start_cursor = blocks["next_cursor"]
            block["children"].extend(blocks['results'])
            if start_cursor is None:
                break  
        
        for child_block in block["children"]:
            c = block_parser(child_block, notion)
            if c is None:
                continue
            content += " " + c
    return content

#Note: a page with no blocks (i.e. empty page with title) will currently return None, None here
def notion_page_parser(page_id: str, notion: "notion_client.client.Client"):
    page = execute_notion_with_retry(notion.pages.retrieve, page_id)
    if page is None:
        return None, None

    start_cursor = None
    all_blocks = []
    while True:
        if start_cursor is None:
            blocks = execute_notion_with_retry(notion.blocks.children.list, page_id)      
        else:
            blocks = execute_notion_with_retry(notion.blocks.children.list, page_id, start_cursor=start_cursor)
        if blocks is None:
            return None, None

        start_cursor = blocks['next_cursor']
        all_blocks.extend(blocks['results'])
        if start_cursor is None:
            break  
    
    content = None
    if page['properties'] is not None and 'title' in page['properties'] and len(page['properties']['title']['title']) > 0:
        content = page['properties']['title']['title'][0]['plain_text']
    elif page['parent']['type'] == 'database_id':
        content = "Database Name: " + get_db_source(page['parent']['database_id'], notion) + " "
        for key in page['properties']:
            property = page['properties'][key]
            if property['type'] == 'title' and len(property['title']) > 0:
                content += "| Datbase entry: " + property['title'][0]['plain_text']
    source = content

    if len(all_blocks) == 0 or source is None:
        return None, None
    
    for block in all_blocks:
        block_content = block_parser(block, notion)
        if block_content is None:
            continue
        content += " " + block_content
    add_id(page_id)
    return content, source

class RowProperty:
    def __init__(self, key, property):
        self.name = key
        self.value = ""
        self.type = property['type']
        if property['type'] == 'select':
            if property['select'] and 'name' in property['select']:
                self.value = property['select']['name']
        elif property['type'] == 'multi_select':
            self.value = ''.join([x['name'] for x in property['multi_select']])
        elif property['type'] == 'title':
            self.value = ''.join([x['plain_text'] for x in property['title']])
        elif property['type'] == 'rich_text':
            self.value = ''.join([x['plain_text'] for x in property['rich_text']])
        elif property['type'] == 'number':
            self.value = property['number']
        elif property['type'] == 'date':
            if property['date'] and 'start' in property['date']:
                self.value = property['date']['start']
        elif property['type'] == 'people':
            self.value = ''.join([x['name'] for x in property['people']])
        elif property['type'] == 'files':
            self.value = ''.join([x['name'] for x in property['files']])
        elif property['type'] == 'checkbox':
            self.value = property['checkbox']  
        elif property['type'] == 'url':
            self.value = property['url']
        elif property['type'] == 'email':
            self.value = property['email']
        elif property['type'] == 'phone_number':
            self.value = property['phone_number']
        # else:
        #     print("Unknown property type: ", property['type'], " for key: ", key, " and value: ", property)   

row_template = '''column-{{index}} name: {{prop.name}} 
column-{{index}} data type: {{prop.type}}  
column-{{index}} value: {{prop.value}}
''' 

class NotionRow:
    def __init__(self, row):
        self.created_time = row["created_time"]
        self.last_edited_time = row["last_edited_time"]
        self.id = row["id"]
        self.properties = [RowProperty(key, value) for key, value in row["properties"].items()]
    
    def serialize(self, database_name, index):
        template = f'''Database name: {database_name}, Created Time: {self.created_time}, Last Edited Time: {self.last_edited_time} \n'''

        for i, prop in enumerate(self.properties):
            t = row_template
            t = t.replace('{{index}}', str(i))
            t = t.replace('{{prop.name}}', str(prop.name))
            t = t.replace('{{prop.type}}', str(prop.type))
            t = t.replace('{{prop.value}}', str(prop.value))
            template += t
            # template += f'''
            # column-{index} name: {prop.name}
            # column-{index} data type: {prop.type}
            # column-{index} value: {prop.value}
            # '''
        
        return template


def get_rows(response, title):
    # get the results from the first page 
    extracted = []
    rows = response["results"]
    for i, row in enumerate(rows):
        clean_row = NotionRow(row)
        extracted.append(clean_row.serialize(title, i))
        # create string that contains all of the properties

    return extracted


def get_db_data(database_id):
    """Get all of the data from a notion database."""
    filter = None
    if database_id == 'de8651d8-a813-4cd2-b803-e21a4dc0871f':
        filter = {
            "and": [
                {
                    "timestamp": "created_time",
                    "created_time": {
                        "on_or_after": "2022-01-01"
                    }
                },
                {
                    "timestamp": "created_time",
                    "created_time": {
                        "on_or_before": "2022-03-31"
                    }
                }
            ]
        }

    response = execute_notion_with_retry(notion.databases.retrieve, database_id)
    if response is None:
        return None, None
    database_title = response["title"][0]["plain_text"]

    response = execute_notion_with_retry(notion.databases.query, database_id, filter=filter)
    if response is None:
        return None, None

    # get the total number of pages
    total_pages = response["has_more"]

    rows = get_rows(response, database_title)
    # if there are more pages, then get the data from the rest of the pages
    if total_pages:
        # get the cursor for the next page
        next_cursor = response["next_cursor"]

        # loop through the rest of the pages
        while next_cursor:
            # get the next page of data
            response = execute_notion_with_retry(notion.databases.query, 
            database_id, start_cursor=next_cursor, filter=filter)
            if response is None:
                return None, None

            r = get_rows(response, database_title)
            rows.extend(r)

            # if there are more pages, then get the data from the rest of the pages
            if response["has_more"]:
                # get the cursor for the next page
                next_cursor = response["next_cursor"]
            else:
                # if there are no more pages, then set the cursor to None
                next_cursor = None

    sources = [database_title] * len(rows)
    add_id(database_id)
    return rows, sources

def text_splitter(text, source):
    chunks = []
    chunk = ""
    naive_split = re.split(r"[,;:\n]", text)
    for split in naive_split:
      if len(chunk) < 3900:
        chunk += split
      else:
        chunks.append(chunk)
        chunk = ""
    chunks.append(chunk)
    for i, chunk in enumerate(chunks):
        if i == 0:
            continue
        chunks[i] = source + " : " + chunk
    return chunks

def text_splitter_db(text, source):
    chunks = []
    chunk = ""
    naive_split = re.split(r'\|', text)
    for split in naive_split:
      if len(chunk) < 6000:
        chunk += split
      else:
        chunks.append(chunk)
        chunk = ""
    chunks.append(chunk)
    for i, chunk in enumerate(chunks):
        if i == 0:
            continue
        chunks[i] = source + " : " + chunk
    return chunks    

#Use notion search api endpoint to get all pages and databases

def process_page(entity_id, notion):
    page_content, source = notion_page_parser(entity_id, notion)
    if page_content == None or source == None:
        return None, None
    page_content = text_splitter(page_content, source)
    return page_content, [source] * len(page_content)

def process_database(entity_id):
    rows, row_sources = get_db_data(entity_id)
    new_rows = []
    new_row_sources = []
    for i, row in enumerate(rows):
        if type(row) is not str:
            print("row is not a string", row, type(row))
            continue
        r = text_splitter_db(row, row_sources[i])
        new_rows.extend(r)
        new_row_sources.extend([row_sources[i]] * len(r))
    return new_rows, new_row_sources

def get_all_pages_and_databases(notion):
    start_cursor = None
    content = []
    sources = []
    while True:
        if start_cursor is None:
            res = execute_notion_with_retry(notion.search)
        else:
            res = execute_notion_with_retry(notion.search, start_cursor=start_cursor)
        
        for entity in res['results']:
            if entity['object'] == 'page':
                page_content, source = notion_page_parser(entity['id'], notion)
                if page_content == None or source == None:
                    continue
                
                page_content = text_splitter(page_content, source)
                content.extend(page_content)
                sources.extend([source] * len(page_content))
            elif entity['object'] == 'database':
                rows, row_sources = get_db_data(entity['id'])

                new_rows = []
                new_row_sources = []
                for i, row in enumerate(rows):
                    if type(row) is not str:
                        print("row is not a string", row, type(row))
                        continue
                    r = text_splitter_db(row, row_sources[i])
                    new_rows.extend(r)
                    new_row_sources.extend([row_sources[i]] * len(r))
            # we don't care about users right now
        
        if res["has_more"] == False or res["next_cursor"] is None:
            break
        start_cursor = res["next_cursor"]
    return content, sources

pages = 1
def get_all_pages_and_databases_async(notion):
    global pages
    start_cursor = None
    content = []
    sources = []

    while True:
        if start_cursor is None:
            res = execute_notion_with_retry(notion.search, page_size=100)
        else:
            res = execute_notion_with_retry(notion.search, start_cursor=start_cursor, page_size=100)
            pages += 1
        
        if res is None or res["results"] is None or res["results"] == []:
            print("Something went very wrong, res is none. Current content", content, start_cursor)
            print("Parsed IDs", parsed_ids)
            print("Parsed Blocks",block_ids)
            break

        allowed_ids = ['148bba51d4b648b5b67661e68ee16d98', '96b6f22292c44ac2a8830ada04a71b57', 'f4a4dfe19c584e35a83036e1a4d62e35']
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_entity = {}
            for entity in res['results']:
                if entity['id'] == 'de8651d8a8134cd2b803e21a4dc0871f':
                    print("hello")
                # this probably includes all pages that are in a database and we want to only use filtered pages
                if entity['object'] == 'page' and entity['parent']['type'] == 'database_id':
                    continue
                elif entity['object'] == 'page' and entity['id'] in allowed_ids:
                    future = executor.submit(process_page, entity['id'], notion)
                    future_to_entity[future] = entity
                elif entity['object'] == 'database':
                    future = executor.submit(process_database, entity['id'])
                    future_to_entity[future] = entity
                # we don't care about users right now
            
            for future in concurrent.futures.as_completed(future_to_entity):
                entity = future_to_entity[future]
                c, s = future.result()
                if c is None or s is None:
                    print("c or s is none", entity['id'])
                    continue
                content.extend(c)
                sources.extend(s)
            
            if res["has_more"] == False or res["next_cursor"] is None:
                break
            start_cursor = res["next_cursor"]
    return content, sources

def get_all_pages_and_databases(notion):
    start_cursor = None
    content = []
    sources = []
    while True:
        if start_cursor is None:
            res = execute_notion_with_retry(notion.search)
        else:
            res = execute_notion_with_retry(notion.search, start_cursor=start_cursor)
        
        for entity in res['results']:
            if entity['object'] == 'page':
                page_content, source = notion_page_parser(entity['id'], notion)
                if page_content == None or source == None:
                    continue
                
                page_content = text_splitter(page_content, source)
                content.extend(page_content)
                sources.extend([source] * len(page_content))
            elif entity['object'] == 'database':
                rows, row_sources = get_db_data(entity['id'])

                new_rows = []
                new_row_sources = []
                for i, row in enumerate(rows):
                    if type(row) is not str:
                        print("row is not a string", row, type(row))
                        continue
                    r = text_splitter_db(row, row_sources[i])
                    new_rows.extend(r)
                    new_row_sources.extend([row_sources[i]] * len(r))
            # we don't care about users right now
        
        if res["has_more"] == False or res["next_cursor"] is None:
            break
        start_cursor = res["next_cursor"]
    return content, sources

print("start time: ", datetime.now())
docs, metadatas = get_all_pages_and_databases_async(notion)
print("end time: ", datetime.now())
print("len(docs): ", len(docs))
print("len(metadatas): ", len(metadatas))
print("count", notion_call_count)
# print("parsed_ids", parsed_ids)
# print("block_ids", block_ids)
# print("docs", docs)
# print("metadata", metadatas)

docs_np = np.array(docs)
metadatas_np = np.array(metadatas)

# Saving the array in a text file
np.save('docs_np_moonchaser_feb22.npy', docs_np) # save
np.save('metadata_np_moonchaser_feb22.npy', metadatas_np) # save




 
    
