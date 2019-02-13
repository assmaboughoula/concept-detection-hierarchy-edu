python3 pdf_concept_miner.py -i Han-Index.pdf -o han_concepts.txt
python3 pdfTextMiner.py (Need to modify strings in the file for input and output file)
python3 iob_tagger_textbook.py -i han_main.txt -o han_tb_iob_tags.txt -c han_concepts.txt