# pdfTextMiner.py
# Python 2.7.6
# For Python 3.x use pdfminer3k module
# This link has useful information on components of the program
# https://euske.github.io/pdfminer/programming.html
# http://denis.papathanasiou.org/posts/2010.08.04.post.html


''' Important classes to remember
PDFParser - fetches data from pdf file
PDFDocument - stores data parsed by PDFParser
PDFPageInterpreter - processes page contents from PDFDocument
PDFDevice - translates processed information from PDFPageInterpreter to whatever you need
PDFResourceManager - Stores shared resources such as fonts or images used by both PDFPageInterpreter and PDFDevice
LAParams - A layout analyzer returns a LTPage object for each page in the PDF document
PDFPageAggregator - Extract the decive to page aggregator to get LT object elements
'''

import os
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
# From PDFInterpreter import both PDFResourceManager and PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
# Import this to raise exception whenever text extraction from PDF is not allowed
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar
from pdfminer.converter import PDFPageAggregator

''' This is what we are trying to do:
1) Transfer information from PDF file to PDF document object. This is done using parser
2) Open the PDF file
3) Parse the file using PDFParser object
4) Assign the parsed content to PDFDocument object
5) Now the information in this PDFDocumet object has to be processed. For this we need
   PDFPageInterpreter, PDFDevice and PDFResourceManager
 6) Finally process the file page by page 
'''
from sklearn.cluster import KMeans	
from statistics import mean
import re
import argparse

concept_re = re.compile(r"([^.,(]*)(?:\.|,|\b|\()") # Make sure excluding periods is okay
NUM_COLS = 2
NUM_SUBCOLS = 3

class TextWrapper():
	def __init__(self, text, x, y, round_dec=None):
		self.text = text
		if round_dec:
			self.x = round(x, round_dec)
			self.y = round(y, round_dec)
		else:
			self.x = x
			self.y = y

def get_pages_textblocks(index_filename):
	password = ""
	extracted_text = ""

	# Open and read the pdf file in binary mode
	fp = open(index_filename, "rb")

	# Create parser object to parse the pdf content
	parser = PDFParser(fp)

	# Store the parsed content in PDFDocument object
	document = PDFDocument(parser, password)

	# Check if document is extractable, if not abort
	if not document.is_extractable:
		raise PDFTextExtractionNotAllowed
		
	# Create PDFResourceManager object that stores shared resources such as fonts or images
	rsrcmgr = PDFResourceManager()

	# set parameters for analysis
	laparams = LAParams(line_margin=.1)

	# Create a PDFDevice object which translates interpreted information into desired format
	# Device needs to be connected to resource manager to store shared resources
	# device = PDFDevice(rsrcmgr)
	# Extract the decive to page aggregator to get LT object elements
	device = PDFPageAggregator(rsrcmgr, laparams=laparams)

	# Create interpreter object to process page content from PDFDocument
	# Interpreter needs to be connected to resource manager for shared resources and device 
	interpreter = PDFPageInterpreter(rsrcmgr, device)

	pages = []
	# Ok now that we have everything to process a pdf document, lets process it page by page
	for page in PDFPage.create_pages(document):
		text_blocks = []
		# As the interpreter processes the page stored in PDFDocument object
		interpreter.process_page(page)
		# The device renders the layout from interpreter
		layout = device.get_result()
		# Out of the many LT objects within layout, we are interested in LTTextBox and LTTextLine
		for lt_obj in layout:
			if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
				tbw = TextWrapper(lt_obj.get_text().strip(), lt_obj.bbox[0], lt_obj.bbox[1])
				text_blocks.append(tbw)
				#print(tbw.text)
				#print(lt_obj.bbox)
		pages.append(text_blocks)
				
	#close the pdf file
	fp.close()
	return pages

def remove_bad_blocks(pages):
	new_pages = []
	for page in pages:
		new_page = []
		for tbw in page:
			# Is the word index or a single letter or a page number
			text = tbw.text
			if not (text.lower() == 'index' or len(text) <= 1 or text.isnumeric()):
				new_page.append(tbw)
		if new_page:
			new_pages.append(new_page)
	return new_pages

def determine_cols(num_cols, text_wrappers):
	# Uses kmeans to cluster x coordinates into columns
	kmeans = KMeans(n_clusters=num_cols, random_state=37)

	x_coords = [[tbw.x] for tbw in text_wrappers]
	kmeans.fit(x_coords)
	labels = kmeans.predict(x_coords)
	
	col_order = list(sorted(enumerate(list(kmeans.cluster_centers_)), key=lambda t: t[1]))
	label_to_col = {col_order[col][0]: col for col in range(num_cols)}
	return zip(text_wrappers, [label_to_col[label] for label in labels])

def flatten_pages(pages):
	text_blocks = []
	for page in pages:
		text_wrappers_and_cols = determine_cols(NUM_COLS, page)

		# Divides TextWrappers Into Columns
		text_wrappers_in_cols = [[] for i in range(NUM_COLS)]
		for tbw, col in text_wrappers_and_cols:
			text_wrappers_in_cols[col].append(tbw)

		# Orders textblocks within a column by y value
		text_wrappers_in_cols = [sorted(col, key=lambda tbw: -tbw.y) for col in text_wrappers_in_cols]

		# Adds new column to all textblocks, normalizes x coordinate since starting x coordinate differs over pages
		for col in text_wrappers_in_cols:
			min_x = min(col, key=lambda tbw: tbw.x).x
			text_blocks.extend([TextWrapper(tbw.text, tbw.x-min_x, tbw.y) for tbw in col])
	return text_blocks

def get_base_concepts(text_blocks):
	concepts = []
	for tb in text_blocks:
		for line in tb.text.split('\n'):
			m = concept_re.match(line)
			if m:
				#print(m.group(1))
				text = m.group(1).strip()
				if text:
					concepts.append(TextWrapper(text, tb.x, tb.y, round_dec=2))
					#print(tb.x)
			else:
				pass
				#print('Not Concept:', lt_obj.get_text())
	return concepts

def concept_generalness(concepts, blacklist=[]):
	def has_letters(s):
		for c in s:
			if c.isalpha():
				return True
		return False

	text_wrappers_and_subcols = determine_cols(NUM_SUBCOLS, concepts)
	cur_general_concept = ''
	cur_specific_concept = ''
	gconcepts = set()

	last_subcol = 0
	for tbw, subcol in text_wrappers_and_subcols:
		if tbw.text in blacklist:
			continue
		#print(tbw.text, subcol, tbw.x)
		if subcol == 0:
			last_subcol = 0
			gconcepts.add(cur_general_concept)
			cur_general_concept = tbw.text
		elif subcol == 1:
			last_subcol = 1
			# We are ignoring specific topics
			#gconcepts.append(last_gconcept)
			#cur_specific_concept = tbw.text
		elif subcol == 2:
			if has_letters(tbw.text) and last_subcol == 0:
				cur_general_concept += (' ' + tbw.text)
	gconcepts.add(cur_general_concept)
	gconcepts.remove('')
	return sorted(list(gconcepts))
	
def main():
	base_path = "."
	parser = argparse.ArgumentParser(description='Extracts concepts from PDF of index.')
	parser.add_argument('--index_filename', '-i')
	parser.add_argument('--concepts_filename', '-o')
	args = parser.parse_args()

	index_filename = os.path.join(base_path + "/" + args.index_filename)
	pages = get_pages_textblocks(index_filename)
	pages = remove_bad_blocks(pages)
	text_blocks = flatten_pages(pages)
	concepts = get_base_concepts(text_blocks)
	gconcepts = concept_generalness(concepts, blacklist=['Numbers and Symbols', 'Zhai'])
	with open(args.concepts_filename, 'w') as f:
		for concept in gconcepts:
			f.write(concept + '\n')

if __name__ == '__main__':
	main()	